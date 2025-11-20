# BigBrother

This project is a cross-platform desktop application designed to help forgetful individuals and those with Alzheimer's by automatically logging daily events from their environment. Using the user’s webcam and microphone, the system will monitor and record important moments – for example, noting where the user placed their keys – and listen to conversations to create reminders, daily summaries, and conversation feeds. The application emphasizes privacy by running entirely locally (no cloud storage) and uses AI for intelligent motion detection, image understanding, and speech analysis.

## Getting Started

This application is packaged with Electron to run as a standalone desktop app.

1.  **Install Dependencies:**
    First, install the necessary Node.js modules and Python packages.
    ```bash
    npm install
    cd Backend
    pip install -r requirements.txt
    cd ..
    ```

2.  **Run the Application:**
    From the root directory, run the following command to launch the application:
    ```bash
    npm run electron:dev
    ```
    This will start the backend server, the frontend, and open the application window.

## Package for Production

To create a distributable desktop application for your operating system (e.g., `.dmg` for macOS, `.exe` for Windows), run the following command from the root directory:

```bash
npm run electron-pack
```

The packaged application will be located in the `dist` folder.

## Development Setup

For developers who want to work on the source code, the frontend and backend can be run separately.

### Backend Setup

1.  **Navigate to the Backend Directory:**
    ```bash
    cd Backend
    ```

2.  **Install Python Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Backend Server:**
    ```bash
    python3 app.py
    ```
    The backend server will be running on [http://localhost:5000](http://localhost:5000).

### Frontend Setup

1.  **Install JavaScript Dependencies:**
    From the root directory of the project:
    ```bash
    npm install
    ```

2.  **Start the Frontend Development Server:**
    ```bash
    npm start
    ```
    The app will open at [http://localhost:3000](http://localhost:3000).

## Technologies Used

### Frontend
- React 18.2.0
- Create React App 5.0.1
- Tailwind CSS

### Backend
- Python 3
- Flask
- OpenCV
- Google Gemini
- SQLite

### Deployment
- Electron
- Electron Builder
- Concurrently

### In Progress
- Fixing Motion Detection - Recording UI/Backend
- Adding way to delete timeline events
- Secret Pivot (TOP SECRET)


