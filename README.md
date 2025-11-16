# BigBrother

This project is a cross-platform web application designed to help forgetful individuals and those with Alzheimer's by automatically logging daily events from their environment. Using the user’s webcam and microphone, the system will monitor and record important moments – for example, noting where the user placed their keys – and listen to conversations to create reminders, daily summaries, and conversation feeds. The application emphasizes privacy by running locally (no cloud storage) and uses AI for intelligent motion detection, image understanding, and speech analysis.

## Getting Started

This project has a separate frontend and backend. You will need to run both for the application to work correctly.

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

### Build for Production

```bash
npm run build
```

This creates an optimized production build in the `build` folder.

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

