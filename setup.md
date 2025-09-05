Steps to run this project

Dependencies required

1. python - https://www.python.org/downloads/
2. ffmpeg (for video processing tasks) - https://www.gyan.dev/ffmpeg/builds/ (extract and set path environment variable to the bin folder)
3. nodejs (for frontend) - https://nodejs.org/en/download/current/?section=1&step=0
4. GPU with cuda installed (optional)

Setup environment

1. Create a virtual environment (preferred) - python -m virtualenv env -> env\Scripts\activate (windows) or source env/bin/activate (linux/mac)
2. Install pytorch (according to your setup) - https://pytorch.org/get-started/locally/
3. pip install -r requirements.txt (install all dependencies)
4. npm install (for frontend)

Start servers (start both)

1. npm run start (frontend) - http://localhost:3000
2. python api.py (backend) - http://localhost:8001
