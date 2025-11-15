# Quick Start Guide - Record and Test

## Step 1: Start the Server

Open a terminal and run:

```bash
cd backend
python3 app.py
```

The server will start on **port 5001** (to avoid AirPlay conflict on macOS).

You should see:
```
 * Running on http://0.0.0.0:5001
```

## Step 2: Record Audio (Easy Way)

Open a **new terminal** and run the interactive script:

```bash
cd backend
python3 record_and_test.py
```

This will give you a menu:
1. **Record audio** - Records and automatically saves as memory node
2. **Ask a question** - Uses Gemini to find relevant memories
3. **List all memories** - See all your recorded memories
4. **Exit**

### Recording Process:
1. Choose option 1
2. Press Enter to start recording
3. **Speak your message** (e.g., "I have a meeting tomorrow at 3pm")
4. Press Enter again to stop
5. Wait for transcription (takes a few seconds)
6. Your memory is saved!

## Step 3: Test Recall

1. Record a few different things (meetings, names, tasks, etc.)
2. Choose option 2 to ask questions
3. Ask natural language questions like:
   - "What did I say about the meeting?"
   - "When is my deadline?"
   - "What's my name?"

## Manual Testing (Using curl)

If you prefer using curl commands:

### Record Audio:
```bash
# Start recording
curl -X POST http://localhost:5001/api/record/start \
  -H "Content-Type: application/json" \
  -d '{"output_path": "my_recording.wav"}'

# Wait a few seconds while speaking, then stop (auto-transcribes)
curl -X POST http://localhost:5001/api/record/stop \
  -H "Content-Type: application/json"
```

### Ask a Question:
```bash
curl -X POST http://localhost:5001/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What did I say about the meeting?"}'
```

### List All Memories:
```bash
curl http://localhost:5001/api/memories
```

## Tips

1. **Speak clearly** - Better audio = better transcription
2. **Record varied content** - Different topics help test recall better
3. **Wait for transcription** - It takes a few seconds after stopping
4. **Check your .env file** - Make sure `GOOGLE_API_KEY` is set

## Troubleshooting

**"Server is not running"**
- Make sure you started `python3 app.py` in another terminal
- Check it's running on port 5001

**"GOOGLE_API_KEY not found"**
- Create a `.env` file in the `backend` directory
- Add: `GOOGLE_API_KEY=your_key_here`

**"No memories found"**
- Record some audio first using option 1
- Check with option 3 to see your memories

**Port already in use**
- The app uses port 5001 by default
- You can change it by setting `PORT=5002 python3 app.py`

