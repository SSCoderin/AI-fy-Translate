// Global variables
let currentVideoId = null;
let currentJobId = null;
let statusCheckInterval = null;
let editingData = null;
let currentAudioPaths = null;

// DOM elements
const uploadForm = document.getElementById("uploadForm");
const videoFileInput = document.getElementById("videoFile");
const selectedFileDiv = document.getElementById("selectedFile");
const fileNameSpan = document.getElementById("fileName");
const fileSizeSpan = document.getElementById("fileSize");
const uploadBtn = document.getElementById("uploadBtn");
const statusSection = document.getElementById("statusSection");
const loadingOverlay = document.getElementById("loadingOverlay");
const editingSection = document.getElementById("editingSection");
const hindiTranscription = document.getElementById("hindiTranscription");
const marathiAudioTranslation = document.getElementById(
  "marathiAudioTranslation"
);
const textTranslationsList = document.getElementById("textTranslationsList");
const saveTranslationsBtn = document.getElementById("saveTranslationsBtn");
const generateFinalBtn = document.getElementById("generateFinalBtn");
const playOriginalBtn = document.getElementById("playOriginalBtn");
const playVoiceBtn = document.getElementById("playVoiceBtn");
const playMusicBtn = document.getElementById("playMusicBtn");

// Status elements
const originalFileNameSpan = document.getElementById("originalFileName");
const videoIdSpan = document.getElementById("videoId");
const jobIdSpan = document.getElementById("jobId");
const statusIcon = document.getElementById("statusIcon");
const statusMain = document.getElementById("statusMain");
const statusSub = document.getElementById("statusSub");
const progressFill = document.getElementById("progressFill");
const checkStatusBtn = document.getElementById("checkStatusBtn");
const downloadBtn = document.getElementById("downloadBtn");
const newTranslationBtn = document.getElementById("newTranslationBtn");
const resultSection = document.getElementById("resultSection");
const outputPath = document.getElementById("outputPath");
const completedAt = document.getElementById("completedAt");
const errorSection = document.getElementById("errorSection");
const errorMessage = document.getElementById("errorMessage");

// Jobs list elements
const refreshJobsBtn = document.getElementById("refreshJobsBtn");
const jobsList = document.getElementById("jobsList");

// Utility functions
function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

function formatDateTime(dateString) {
  if (!dateString) return "-";
  return new Date(dateString).toLocaleString();
}

function showLoading() {
  loadingOverlay.style.display = "flex";
}

function hideLoading() {
  loadingOverlay.style.display = "none";
}

function showNotification(message, type = "info") {
  // Simple notification system
  const notification = document.createElement("div");
  notification.className = `notification notification-${type}`;
  notification.textContent = message;

  // Style the notification
  Object.assign(notification.style, {
    position: "fixed",
    top: "20px",
    right: "20px",
    padding: "1rem 1.5rem",
    borderRadius: "0.5rem",
    color: "white",
    fontWeight: "500",
    zIndex: "1001",
    maxWidth: "400px",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.2)",
    transform: "translateX(100%)",
    transition: "transform 0.3s ease",
  });

  // Set background color based on type
  const colors = {
    success: "#48bb78",
    error: "#e53e3e",
    warning: "#ed8936",
    info: "#4299e1",
  };
  notification.style.background = colors[type] || colors.info;

  document.body.appendChild(notification);

  // Animate in
  setTimeout(() => {
    notification.style.transform = "translateX(0)";
  }, 100);

  // Remove after delay
  setTimeout(() => {
    notification.style.transform = "translateX(100%)";
    setTimeout(() => {
      document.body.removeChild(notification);
    }, 300);
  }, 4000);
}

// File input handling
videoFileInput.addEventListener("change", function (e) {
  const file = e.target.files[0];
  if (file) {
    fileNameSpan.textContent = file.name;
    fileSizeSpan.textContent = formatFileSize(file.size);
    selectedFileDiv.style.display = "block";

    // Validate file size (100MB limit)
    if (file.size > 100 * 1024 * 1024) {
      showNotification("File too large! Maximum size is 100MB.", "error");
      videoFileInput.value = "";
      selectedFileDiv.style.display = "none";
      return;
    }

    // Validate file type
    const allowedTypes = [
      "video/mp4",
      "video/avi",
      "video/mov",
      "video/wmv",
      "video/webm",
    ];
    if (!allowedTypes.includes(file.type)) {
      showNotification(
        "Invalid file type! Please upload a video file.",
        "error"
      );
      videoFileInput.value = "";
      selectedFileDiv.style.display = "none";
      return;
    }
  } else {
    selectedFileDiv.style.display = "none";
  }
});

// Form submission
uploadForm.addEventListener("submit", async function (e) {
  e.preventDefault();

  const file = videoFileInput.files[0];
  if (!file) {
    showNotification("Please select a video file first!", "error");
    return;
  }

  const formData = new FormData();
  formData.append("video", file);

  // Add moving text checkbox value
  const movingTextCheckbox = document.getElementById("movingText");
  formData.append("movingText", movingTextCheckbox.checked);

  // Add advanced OCR checkbox value
  const advancedOcrCheckbox = document.getElementById("advancedOcr");
  formData.append("advancedOcr", advancedOcrCheckbox.checked);

  // Update UI
  uploadBtn.disabled = true;
  uploadBtn.querySelector(".btn-text").textContent = "Uploading...";
  uploadBtn.querySelector(".spinner").style.display = "block";
  showLoading();

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (response.ok && result.success) {
      // Store current job info
      currentVideoId = result.videoId;
      currentJobId = result.jobId;

      // Update status section
      originalFileNameSpan.textContent = result.originalFilename;
      videoIdSpan.textContent = result.videoId;
      jobIdSpan.textContent = result.jobId;

      // Show status section
      statusSection.style.display = "block";
      statusSection.scrollIntoView({ behavior: "smooth" });

      // Start checking status
      updateStatus(result.status, "Translation started successfully!");
      startStatusChecking();

      // Reset form
      uploadForm.reset();
      selectedFileDiv.style.display = "none";

      showNotification(
        "Video uploaded successfully! Translation started.",
        "success"
      );
    } else {
      throw new Error(result.error || "Upload failed");
    }
  } catch (error) {
    console.error("Upload error:", error);
    showNotification(`Upload failed: ${error.message}`, "error");
  } finally {
    // Reset upload button
    uploadBtn.disabled = false;
    uploadBtn.querySelector(".btn-text").textContent = "Start Translation";
    uploadBtn.querySelector(".spinner").style.display = "none";
    hideLoading();
  }
});

// Status checking
async function checkJobStatus(videoId = currentVideoId) {
  if (!videoId) return;

  try {
    const response = await fetch(`/status/${videoId}`);
    const result = await response.json();

    if (response.ok && result.success) {
      updateStatusDisplay(result);
      return result;
    } else {
      throw new Error(result.error || "Status check failed");
    }
  } catch (error) {
    console.error("Status check error:", error);
    showNotification(`Status check failed: ${error.message}`, "error");
    return null;
  }
}

function updateStatusDisplay(statusData) {
  const status = statusData.status;
  const progress = statusData.progress || "Processing...";

  // Update status display
  updateStatus(status, progress);

  // Update progress bar
  let progressPercent = 0;
  switch (status) {
    case "queued":
      progressPercent = 10;
      break;
    case "processing":
      progressPercent = 50;
      break;
    case "completed":
      progressPercent = 100;
      break;
    case "failed":
      progressPercent = 0;
      break;
  }
  progressFill.style.width = progressPercent + "%";

  // Show/hide sections based on status
  if (status === "completed") {
    // Show result section
    resultSection.style.display = "block";
    outputPath.textContent = statusData.outputVideoPath || "N/A";
    completedAt.textContent = formatDateTime(statusData.completedAt);
    downloadBtn.style.display = "inline-flex";

    // Hide error section
    errorSection.style.display = "none";

    // Stop status checking
    stopStatusChecking();
    showEditingButton();

    showNotification("Translation completed successfully!", "success");
  } else if (status === "failed") {
    // Show error section
    errorSection.style.display = "block";
    errorMessage.textContent = statusData.error || "Unknown error occurred";

    // Hide result section
    resultSection.style.display = "none";
    downloadBtn.style.display = "none";

    // Stop status checking
    stopStatusChecking();

    showNotification("Translation failed!", "error");
  } else {
    // Hide both sections for processing states
    resultSection.style.display = "none";
    errorSection.style.display = "none";
    downloadBtn.style.display = "none";
  }
}
function showEditingButton() {
    // Add editing button to action buttons if it doesn't exist
    const actionButtons = document.querySelector('.action-buttons');
    
    if (!document.getElementById('startEditingBtn')) {
        const editingBtn = document.createElement('button');
        editingBtn.id = 'startEditingBtn';
        editingBtn.className = 'edit-btn';
        editingBtn.innerHTML = '✏️ Edit Translations';
        editingBtn.onclick = () => separateAudioAndShowEditor(currentVideoId);
        
        // Insert before download button
        actionButtons.insertBefore(editingBtn, downloadBtn);
    }
}

function updateStatus(status, message) {
  // Update icon and text based on status
  const statusConfig = {
    queued: { icon: "⏳", main: "Queued", className: "status-queued" },
    processing: {
      icon: "🔄",
      main: "Processing",
      className: "status-processing",
    },
    completed: { icon: "✅", main: "Completed", className: "status-completed" },
    failed: { icon: "❌", main: "Failed", className: "status-failed" },
  };

  const config = statusConfig[status] || statusConfig.processing;

  statusIcon.textContent = config.icon;
  statusIcon.className = "status-icon " + config.className;
  statusMain.textContent = config.main;
  statusSub.textContent = message;
}

function startStatusChecking() {
  if (statusCheckInterval) {
    clearInterval(statusCheckInterval);
  }

  // Check status every 5 seconds
  statusCheckInterval = setInterval(async () => {
    const result = await checkJobStatus();
    if (
      result &&
      (result.status === "completed" || result.status === "failed")
    ) {
      stopStatusChecking();
    }
  }, 5000);
}

function stopStatusChecking() {
  if (statusCheckInterval) {
    clearInterval(statusCheckInterval);
    statusCheckInterval = null;
  }
}

// Manual status check
checkStatusBtn.addEventListener("click", async function () {
  checkStatusBtn.disabled = true;
  checkStatusBtn.innerHTML = "🔄 Checking...";

  await checkJobStatus();

  checkStatusBtn.disabled = false;
  checkStatusBtn.innerHTML = "🔄 Check Status";
});

// Download button
downloadBtn.addEventListener("click", async function () {
  if (!currentVideoId) return;

  try {
    const response = await fetch(`/download/${currentVideoId}`);

    if (response.ok) {
      // Create download link
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `translated_${originalFileNameSpan.textContent}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      showNotification("Download started!", "success");
    } else {
      const error = await response.json();
      throw new Error(error.error || "Download failed");
    }
  } catch (error) {
    console.error("Download error:", error);
    showNotification(`Download failed: ${error.message}`, "error");
  }
});

// New translation button
newTranslationBtn.addEventListener("click", function () {
  // Reset everything
  currentVideoId = null;
  currentJobId = null;
  stopStatusChecking();

  // Hide status section
  statusSection.style.display = "none";

  // Reset form
  uploadForm.reset();
  selectedFileDiv.style.display = "none";

  // Scroll to top
  window.scrollTo({ top: 0, behavior: "smooth" });

  showNotification("Ready for new translation!", "info");
});

// Jobs list management
async function refreshJobsList() {
  try {
    const response = await fetch("/jobs");
    const result = await response.json();

    if (response.ok && result.success) {
      displayJobsList(result.jobs);
    } else {
      throw new Error(result.error || "Failed to fetch jobs");
    }
  } catch (error) {
    console.error("Jobs fetch error:", error);
    showNotification(`Failed to fetch jobs: ${error.message}`, "error");
  }
}

function displayJobsList(jobs) {
  if (!jobs || jobs.length === 0) {
    jobsList.innerHTML =
      '<div class="no-jobs">No translation jobs yet. Upload a video to get started!</div>';
    return;
  }

  const jobsHtml = jobs
    .map(
      (job) => `
        <div class="job-item">
            <div class="job-info-item">
                <div class="job-name">${job.originalFilename}</div>
                <div class="job-status status-${job.status}">
                    Status: ${job.status} ${
        job.progress ? `- ${job.progress}` : ""
      }
                </div>
                <div class="job-id">Video ID: ${job.videoId}</div>
            </div>
            <div class="job-actions">
                <button class="job-action-btn" onclick="loadJobStatus('${
                  job.videoId
                }')">
                    View Status
                </button>
                ${
                  job.status === "completed"
                    ? `<button class="job-action-btn" onclick="downloadJob('${job.videoId}')">Download</button>`
                    : ""
                }
            </div>
        </div>
    `
    )
    .join("");

  jobsList.innerHTML = jobsHtml;
}
async function separateAudioAndShowEditor(videoId) {
  try {
    showNotification("Separating audio...", "info");

    // Step 1: Separate audio
    const separateResponse = await fetch(`/separate-audio/${videoId}`, {
      method: "POST",
    });

    const separateResult = await separateResponse.json();

    if (!separateResponse.ok || !separateResult.voice_audio_path) {
      throw new Error(separateResult.detail || "Audio separation failed");
    }

    currentAudioPaths = {
      voice: separateResult.voice_audio_path,
      music: separateResult.music_audio_path,
    };

    // Step 2: Get editable translations
    const translationsResponse = await fetch(
      `/editable-translations/${videoId}`
    );
    const translationsResult = await translationsResponse.json();

    if (!translationsResponse.ok) {
      throw new Error(
        translationsResult.detail || "Failed to get translations"
      );
    }

    editingData = translationsResult;

    // Step 3: Show editing interface
    showEditingInterface(translationsResult);
    showNotification("Ready for editing!", "success");
  } catch (error) {
    console.error("Error in audio separation and editor setup:", error);
    showNotification(`Failed to setup editor: ${error.message}`, "error");
  }
}

function showEditingInterface(data) {
  // Populate audio translations
  hindiTranscription.value = data.audio_translations.hindi_transcription || "";
  marathiAudioTranslation.value =
    data.audio_translations.marathi_translation || "";

  // Populate text translations
  textTranslationsList.innerHTML = "";

  data.text_translations.forEach((item, index) => {
    const translationItem = document.createElement("div");
    translationItem.className = "translation-item";
    translationItem.innerHTML = `
            <div class="translation-pair">
                <div class="hindi-text">
                    <label>Hindi Text:</label>
                    <input type="text" value="${item.hindi_text}" readonly>
                </div>
                <div class="marathi-text">
                    <label>Marathi Translation:</label>
                    <input type="text" value="${item.marathi_translation}" 
                           data-index="${index}" class="marathi-edit-input">
                </div>
                <div class="bbox-info">
                    Position: [${item.bbox.join(", ")}]
                </div>
            </div>
        `;
    textTranslationsList.appendChild(translationItem);
  });

  // Show editing section
  editingSection.style.display = "block";
  editingSection.scrollIntoView({ behavior: "smooth" });
}

async function saveTranslations() {
  try {
    if (!editingData) {
      throw new Error("No editing data available");
    }

    // Collect edited translations
    const audioTranslation = marathiAudioTranslation.value.trim();

    const textTranslations = [];
    const editInputs = document.querySelectorAll(".marathi-edit-input");

    editInputs.forEach((input, index) => {
      if (index < editingData.text_translations.length) {
        const originalItem = editingData.text_translations[index];
        textTranslations.push({
          hindi_text: originalItem.hindi_text,
          marathi_translation: input.value.trim(),
          bbox: originalItem.bbox,
        });
      }
    });

    // Save changes
    const response = await fetch("/update-translations", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        job_id: currentVideoId,
        audio_translation: audioTranslation,
        text_translations: textTranslations,
      }),
    });

    const result = await response.json();

    if (response.ok) {
      showNotification("Translations saved successfully!", "success");

      // Enable generate final button
      generateFinalBtn.disabled = false;
      generateFinalBtn.style.opacity = "1";
    } else {
      throw new Error(result.detail || "Failed to save translations");
    }
  } catch (error) {
    console.error("Error saving translations:", error);
    showNotification(`Failed to save: ${error.message}`, "error");
  }
}

async function generateFinalVideo() {
  try {
    if (!currentVideoId) {
      throw new Error("No video ID available");
    }

    // Start final video generation
    const response = await fetch(`/generate-final-video/${currentVideoId}`, {
      method: "POST",
    });

    const result = await response.json();

    if (response.ok) {
      // Show final status section
      const finalStatus = document.getElementById("finalStatus");
      finalStatus.style.display = "block";

      // Start checking final status
      startFinalStatusChecking();

      showNotification("Final video generation started!", "success");
    } else {
      throw new Error(
        result.detail || "Failed to start final video generation"
      );
    }
  } catch (error) {
    console.error("Error generating final video:", error);
    showNotification(`Failed to generate: ${error.message}`, "error");
  }
}

function startFinalStatusChecking() {
  const finalStatusInterval = setInterval(async () => {
    try {
      const response = await fetch(`/status/${currentVideoId}`);
      const result = await response.json();

      if (response.ok && result.success) {
        updateFinalStatus(result.status, result.progress);

        if (result.status === "final_completed") {
          clearInterval(finalStatusInterval);
          showFinalVideoReady(result.video_path);
        } else if (result.status === "failed") {
          clearInterval(finalStatusInterval);
          showFinalError(result.error);
        }
      }
    } catch (error) {
      console.error("Error checking final status:", error);
    }
  }, 3000);
}

function updateFinalStatus(status, progress) {
  const finalStatusIcon = document.getElementById("finalStatusIcon");
  const finalStatusMain = document.getElementById("finalStatusMain");
  const finalStatusSub = document.getElementById("finalStatusSub");

  const statusConfig = {
    generating_final: { icon: "⚙️", main: "Generating Final Video" },
    final_completed: { icon: "✅", main: "Final Video Ready!" },
    failed: { icon: "❌", main: "Generation Failed" },
  };

  const config = statusConfig[status] || statusConfig["generating_final"];

  finalStatusIcon.textContent = config.icon;
  finalStatusMain.textContent = config.main;
  finalStatusSub.textContent = progress || "";
}

function showFinalVideoReady(videoPath) {
  // Add final download button
  const finalActions = document.querySelector(".editing-actions");

  const finalDownloadBtn = document.createElement("button");
  finalDownloadBtn.className = "download-btn";
  finalDownloadBtn.innerHTML = "📥 Download Final Video";
  finalDownloadBtn.onclick = () => downloadFinalVideo();

  finalActions.appendChild(finalDownloadBtn);

  showNotification("Final video is ready for download!", "success");
}

async function downloadFinalVideo() {
  try {
    const response = await fetch(`/download/${currentVideoId}`);

    if (response.ok) {
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `final_translated_video_${currentVideoId}.mp4`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      showNotification("Final video download started!", "success");
    } else {
      const error = await response.json();
      throw new Error(error.error || "Download failed");
    }
  } catch (error) {
    console.error("Download error:", error);
    showNotification(`Download failed: ${error.message}`, "error");
  }
}

// Audio playback functions
function playAudio(audioPath) {
  // Simple audio playback (you might want to use a more robust audio player)
  const audio = new Audio(`/${audioPath}`);
  audio.play().catch((error) => {
    console.error("Audio playback failed:", error);
    showNotification("Audio playback failed", "error");
  });
}

// Global functions for job actions
window.loadJobStatus = async function (videoId) {
  currentVideoId = videoId;

  // Check status
  const result = await checkJobStatus(videoId);
  if (result) {
    // Update all job info
    originalFileNameSpan.textContent = result.originalFilename;
    videoIdSpan.textContent = result.videoId;
    jobIdSpan.textContent = result.jobId;

    // Show status section
    statusSection.style.display = "block";
    statusSection.scrollIntoView({ behavior: "smooth" });

    // Start status checking if still processing
    if (result.status === "processing" || result.status === "queued") {
      startStatusChecking();
    }
  }
};

window.downloadJob = async function (videoId) {
  try {
    const response = await fetch(`/download/${videoId}`);

    if (response.ok) {
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `translated_video_${videoId}.mp4`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      showNotification("Download started!", "success");
    } else {
      const error = await response.json();
      throw new Error(error.error || "Download failed");
    }
  } catch (error) {
    console.error("Download error:", error);
    showNotification(`Download failed: ${error.message}`, "error");
  }
};

// Refresh jobs button
refreshJobsBtn.addEventListener("click", refreshJobsList);

// Initialize page
document.addEventListener("DOMContentLoaded", function () {
  // Load jobs list on page load
  refreshJobsList();

  // Check if there's a current job in progress
  if (currentVideoId) {
    startStatusChecking();
  }
   if (saveTranslationsBtn) {
        saveTranslationsBtn.addEventListener('click', saveTranslations);
    }
    
    if (generateFinalBtn) {
        generateFinalBtn.addEventListener('click', generateFinalVideo);
        generateFinalBtn.disabled = true; // Initially disabled until translations are saved
        generateFinalBtn.style.opacity = '0.5';
    }
    
    if (playOriginalBtn) {
        playOriginalBtn.addEventListener('click', () => {
            // Play original audio
            const originalAudio = `audios/audio_${currentVideoId}.wav`;
            playAudio(originalAudio);
        });
    }
    
    if (playVoiceBtn) {
        playVoiceBtn.addEventListener('click', () => {
            if (currentAudioPaths && currentAudioPaths.voice) {
                playAudio(currentAudioPaths.voice);
            }
        });
    }
    
    if (playMusicBtn) {
        playMusicBtn.addEventListener('click', () => {
            if (currentAudioPaths && currentAudioPaths.music) {
                playAudio(currentAudioPaths.music);
            }
        });
    }
});

// Cleanup on page unload
window.addEventListener("beforeunload", function () {
  stopStatusChecking();
});
