const express = require('express');
const multer = require('multer');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const path = require('path');
const fs = require('fs');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;
const API_BASE_URL = process.env.API_BASE_URL || "http://localhost:8001";
// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Ensure upload directory exists
const uploadDir = 'uploads';
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, uploadDir);
    },
    filename: function (req, file, cb) {
        const uniqueId = uuidv4();
        const extension = path.extname(file.originalname);
        const filename = `video_${uniqueId}${extension}`;
        cb(null, filename);
    }
});

// File filter to only allow video files
const fileFilter = (req, file, cb) => {
    const allowedMimes = [
        'video/mp4',
        'video/avi', 
        'video/mov',
        'video/wmv',
        'video/flv',
        'video/webm'
    ];
    
    if (allowedMimes.includes(file.mimetype)) {
        cb(null, true);
    } else {
        cb(new Error('Only video files are allowed!'), false);
    }
};

const upload = multer({ 
    storage: storage,
    fileFilter: fileFilter,
    limits: {
        fileSize: 100 * 1024 * 1024 // 100MB limit
    }
});

// In-memory job tracking (in production, use a database)
const jobs = new Map();

// Routes

// Serve the main page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Upload video and start translation
app.post('/upload', upload.single('video'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No video file uploaded' });
        }

        const videoPath = req.file.path;
        const videoId = path.parse(req.file.filename).name.replace('video_', ''); // Extract UUID only
        
        console.log(`📹 Video uploaded: ${req.file.filename}`);
        console.log(`📁 Saved to: ${videoPath}`);

        // Get moving text flag from request (default to false)
        const movingText = req.body.movingText === 'true' || req.body.movingText === true;
        
        // Get advanced OCR flag from request (default to true)
        const advancedOcr = req.body.advancedOcr === 'true' || req.body.advancedOcr === true;
        
        console.log(`🔄 Moving text mode: ${movingText ? 'ON' : 'OFF'}`);
        console.log(`🔍 Advanced OCR mode: ${advancedOcr ? 'ON' : 'OFF'}`);
        
        // Start translation job via FastAPI using the same UUID as job_id
        const translationResponse = await axios.post(`${API_BASE_URL}/translate`, {
            video_path: videoPath,
            job_id: videoId,  // Use the same UUID from upload for consistent naming
            moving_text: movingText,  // Pass moving text flag to FastAPI
            advanced_ocr: advancedOcr  // Pass advanced OCR flag to FastAPI
        });

        const jobData = {
            videoId: videoId,
            originalFilename: req.file.originalname,
            videoPath: videoPath,
            jobId: translationResponse.data.job_id,
            status: translationResponse.data.status,
            startedAt: translationResponse.data.started_at,
            message: translationResponse.data.message
        };

        // Store job info
        jobs.set(videoId, jobData);

        res.json({
            success: true,
            videoId: videoId,
            jobId: translationResponse.data.job_id,
            status: translationResponse.data.status,
            message: 'Video uploaded and translation started!',
            originalFilename: req.file.originalname
        });

    } catch (error) {
        console.error('Upload error:', error);
        
        // Clean up uploaded file if translation failed
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }

        res.status(500).json({ 
            error: 'Failed to start translation',
            details: error.response?.data?.detail || error.message
        });
    }
});

// Check job status
app.get('/status/:videoId', async (req, res) => {
    try {
        const videoId = req.params.videoId;
        const jobData = jobs.get(videoId);

        if (!jobData) {
            return res.status(404).json({ error: 'Job not found' });
        }

        // Get latest status from FastAPI
        const statusResponse = await axios.get(`${API_BASE_URL}/status/${jobData.jobId}`);
        
        // Update local job data
        jobData.status = statusResponse.data.status;
        jobData.progress = statusResponse.data.progress;
        jobData.completedAt = statusResponse.data.completed_at;
        jobData.error = statusResponse.data.error;
        jobData.outputVideoPath = statusResponse.data.video_path;

        jobs.set(videoId, jobData);

        res.json({
            success: true,
            videoId: videoId,
            jobId: jobData.jobId,
            originalFilename: jobData.originalFilename,
            status: statusResponse.data.status,
            progress: statusResponse.data.progress,
            startedAt: statusResponse.data.started_at,
            completedAt: statusResponse.data.completed_at,
            outputVideoPath: statusResponse.data.video_path,
            error: statusResponse.data.error
        });

    } catch (error) {
        console.error('Status check error:', error);
        res.status(500).json({ 
            error: 'Failed to check status',
            details: error.response?.data?.detail || error.message
        });
    }
});

// Download translated video
app.get('/download/:videoId', async (req, res) => {
    try {
        const videoId = req.params.videoId;
        const jobData = jobs.get(videoId);

        if (!jobData) {
            return res.status(404).json({ error: 'Job not found' });
        }

        if (jobData.status !== 'completed') {
            return res.status(400).json({ 
                error: 'Translation not completed yet',
                status: jobData.status 
            });
        }

        // Proxy download from FastAPI
        const downloadResponse = await axios.get(`${API_BASE_URL}/download/${jobData.jobId}`, {
            responseType: 'stream'
        });

        // Set headers for file download
        res.setHeader('Content-Type', 'video/mp4');
        res.setHeader('Content-Disposition', `attachment; filename="translated_${jobData.originalFilename}"`);

        // Pipe the video stream to response
        downloadResponse.data.pipe(res);

    } catch (error) {
        console.error('Download error:', error);
        res.status(500).json({ 
            error: 'Failed to download video',
            details: error.response?.data?.detail || error.message
        });
    }
});

// Get all jobs
app.get('/jobs', (req, res) => {
    const allJobs = Array.from(jobs.entries()).map(([videoId, data]) => ({
        videoId,
        ...data
    }));

    res.json({
        success: true,
        jobs: allJobs,
        total: allJobs.length
    });
});

// Delete job and cleanup files
app.delete('/jobs/:videoId', async (req, res) => {
    try {
        const videoId = req.params.videoId;
        const jobData = jobs.get(videoId);

        if (!jobData) {
            return res.status(404).json({ error: 'Job not found' });
        }

        // Delete from FastAPI backend
        try {
            await axios.delete(`${API_BASE_URL}/jobs/${jobData.jobId}`);
        } catch (apiError) {
            console.warn('Failed to delete from API:', apiError.message);
        }

        // Clean up uploaded file
        if (jobData.videoPath && fs.existsSync(jobData.videoPath)) {
            fs.unlinkSync(jobData.videoPath);
        }

        // Remove from local storage
        jobs.delete(videoId);

        res.json({
            success: true,
            message: 'Job deleted successfully'
        });

    } catch (error) {
        console.error('Delete error:', error);
        res.status(500).json({ 
            error: 'Failed to delete job',
            details: error.message
        });
    }
});

// Health check
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        activeJobs: jobs.size
    });
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('Server error:', error);
    
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ error: 'File too large. Maximum size is 100MB.' });
        }
    }
    
    res.status(500).json({ 
        error: 'Internal server error',
        message: error.message 
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Frontend server running on http://localhost:${PORT}`);
    console.log(`📡 Connected to API at ${API_BASE_URL}`);
    console.log(`📁 Upload directory: ${path.resolve(uploadDir)}`);
});

module.exports = app;