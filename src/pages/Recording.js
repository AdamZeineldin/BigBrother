import React, { useState } from "react";
import CameraRecorder from "../components/CameraRecorder";
import Timeline from "../components/Timeline";
import Chat from "../components/Chat";

const eventTitles = [
  "event1",
  "event2",
  "event3",
  "event4",
  "event5",
  "event6",
  "event7",
  "event8",
  "event9",
  "event10",
];

function Recording() {
  const [isRecording, setIsRecording] = useState(false);
  const [events, setEvents] = useState([]);
  const [eventIdCounter, setEventIdCounter] = useState(0);
  const [index, setIndex] = useState(0);

  const addEvent = () => {
    if (index >= eventTitles.length) {
      return;
    }
    const newEvent = {
      id: eventIdCounter,
      title: eventTitles[index],
      timestamp: new Date(),
    };
    setEvents((prev) => [...prev, newEvent]);
    setEventIdCounter((prev) => prev + 1);
    setIndex((prev) => prev + 1);
  };

  const handleStartRecording = () => {
    setIsRecording(true);
  };

  const handleStopRecording = () => {
    setIsRecording(false);
  };

  const handleRecordingStart = () => {
    console.log("Recording started");
  };

  const handleRecordingStop = (blob) => {
    console.log("Recording stopped", blob);
  };

  return (
    <main className="flex-1 bg-primary-50 flex flex-col items-center justify-center p-8 relative">
      <div className="flex w-full max-w-7xl gap-8 mb-8">
        <div className="flex-1 flex flex-col items-center justify-center">
          <CameraRecorder
            onRecordingStart={handleRecordingStart}
            onRecordingStop={handleRecordingStop}
            isRecording={isRecording}
          />
          <div className="mt-6 flex space-x-4">
            <button
              onClick={handleStartRecording}
              disabled={isRecording}
              className={`px-6 py-3 rounded-lg font-medium transition-colors duration-200 shadow-sm ${
                isRecording
                  ? "bg-gray-400 text-gray-200 cursor-not-allowed"
                  : "bg-primary-600 text-white hover:bg-primary-700"
              }`}
            >
              Start Recording
            </button>
            <button
              onClick={handleStopRecording}
              disabled={!isRecording}
              className={`px-6 py-3 rounded-lg font-medium transition-colors duration-200 shadow-sm ${
                !isRecording
                  ? "bg-gray-400 text-gray-200 cursor-not-allowed"
                  : "bg-red-600 text-white hover:bg-red-700"
              }`}
            >
              Stop Recording
            </button>
          </div>
        </div>
        <div className="w-96 flex flex-col items-start min-h-0 border-2 p-4">
          <div className="w-full">
            <Timeline events={events} />
          </div>

          <button
            onClick={addEvent}
            disabled={index >= eventTitles.length}
            className={`mt-4 px-4 py-2 rounded-lg font-medium transition-colors duration-200 shadow-sm text-sm ${
              index >= eventTitles.length
                ? "bg-gray-400 text-gray-200 cursor-not-allowed"
                : "bg-primary-600 text-white hover:bg-primary-700"
            }`}
          >
            Add Event
          </button>
        </div>
      </div>

      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 w-full max-w-5xl px-8">
        <Chat onSendMessage={(message) => console.log("Message:", message)} />
      </div>
    </main>
  );
}

export default Recording;
