import React, { useState, useEffect } from "react";
import CameraRecorder from "../components/CameraRecorder";
import Timeline from "../components/Timeline";
import Chat from "../components/Chat";

const API_BASE_URL =
  process.env.REACT_APP_API_URL || "http://localhost:5000/api";

function Recording() {
  const [isRecording, setIsRecording] = useState(false);
  const [motionDetected, setMotionDetected] = useState(false);
  const [isCurrentlyRecording, setIsCurrentlyRecording] = useState(false);
  const [events, setEvents] = useState([]);
  const [selectedEvent, setSelectedEvent] = useState(null);

  // Fetch memory nodes and convert them to events
  const fetchMemoryNodes = async () => {
    try {
      // First, cleanup orphaned memory nodes
      try {
        await fetch(`${API_BASE_URL}/memory-nodes/cleanup`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        });
      } catch (cleanupError) {
        // Silently fail cleanup - not critical
        console.debug("Cleanup failed (non-critical):", cleanupError);
      }

      const response = await fetch(
        `${API_BASE_URL}/memory-nodes?file_type=recording`,
        {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      if (!response.ok) {
        console.error("Failed to fetch memory nodes:", response.status);
        return;
      }

      const data = await response.json();
      const memoryNodes = data.memory_nodes || [];

      // Convert memory nodes to events
      const convertedEvents = memoryNodes.map((node) => {
        let metadata = {};
        try {
          metadata =
            typeof node.metadata === "string"
              ? JSON.parse(node.metadata)
              : node.metadata || {};
        } catch (e) {
          console.error("Error parsing metadata:", e);
        }

        // Check if summary is loading
        const summaryText = metadata.summary;
        const isSummaryLoading =
          summaryText === "Loading Summary..." ||
          summaryText === null ||
          !summaryText ||
          summaryText.trim() === "";
        const fullSummary = isSummaryLoading
          ? "Loading Summary..."
          : summaryText || "No summary available";

        // Use title from metadata if available, otherwise generate from timestamp
        let eventTitle = metadata.title;

        if (!eventTitle || eventTitle.trim() === "") {
          // Format timestamp as title (e.g., "11/15 11:52 PM")
          const timestamp = node.timestamp || new Date().toISOString();

          // Ensure timestamp is treated as UTC - add 'Z' if not present
          let utcTimestamp = timestamp;
          if (
            !utcTimestamp.endsWith("Z") &&
            !utcTimestamp.includes("+") &&
            !utcTimestamp.includes("-", 10)
          ) {
            // If no timezone info, assume it's UTC and add 'Z'
            utcTimestamp = utcTimestamp.replace(/\.\d{3,6}/, "") + "Z";
          }

          const date = new Date(utcTimestamp);

          // Format as EST/EDT for title
          const formatter = new Intl.DateTimeFormat("en-US", {
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            hour12: true,
            timeZone: "America/New_York",
          });

          // Use the formatter directly to get the correct timezone-adjusted string
          eventTitle = formatter.format(date);
        }

        return {
          id: node.id,
          title: eventTitle,
          timestamp: node.timestamp || new Date().toISOString(),
          summary: fullSummary,
          transcript: metadata.transcript || null,
          video_path: metadata.video_path || node.file_path,
          audio_path: metadata.audio_path || null,
          transcript_path: metadata.transcript_path || null,
          objects_detected: metadata.objects_detected || [],
        };
      });

      // Sort by timestamp (oldest first, so newest appears at bottom)
      convertedEvents.sort((a, b) => {
        const dateA = new Date(a.timestamp);
        const dateB = new Date(b.timestamp);
        return dateA - dateB;
      });

      setEvents(convertedEvents);
    } catch (error) {
      console.error("Error fetching memory nodes:", error);
    }
  };

  const handleEventClick = (event) => {
    setSelectedEvent(event);
  };

  const closeEventModal = () => {
    setSelectedEvent(null);
  };

  // Helper function to format summary text with proper line breaks
  const formatSummary = (summary) => {
    if (!summary || summary === "Loading Summary...") {
      return summary;
    }

    let formatted = summary;

    // Ensure there's a blank line before **Key sections
    // Match patterns like **Key Objects:** or **Key People:** or **Key Actions:**
    // Replace any occurrence of **Key sections that don't start on a new line
    formatted = formatted.replace(/([^\n])(\*\*Key\s+[^*]+:\*\*)/g, "$1\n\n$2");

    // Ensure each **Key section starts at the beginning of a line
    formatted = formatted.replace(/\n(\*\*Key\s+[^*]+:\*\*)/g, "\n\n$1");

    // Clean up any triple newlines (more than 2 consecutive newlines)
    formatted = formatted.replace(/\n{3,}/g, "\n\n");

    // Trim leading/trailing whitespace but preserve structure
    formatted = formatted.trim();

    return formatted;
  };

  // Helper function to convert file path to API URL (for videos, audio, images, transcripts)
  const getFileUrl = (filePath) => {
    if (!filePath) return null;

    // Extract relative path from data directory
    // Paths can be:
    // - Absolute: "/Users/victor/Projects/BigBrother/Backend/data/recordings/file.mp4"
    // - Relative to Backend: "Backend/data/recordings/file.mp4"
    // - Already relative: "recordings/file.mp4"
    let relativePath = filePath;

    // If it contains "data/", extract everything after "data/"
    if (filePath.includes("data/")) {
      const dataIndex = filePath.indexOf("data/");
      relativePath = filePath.substring(dataIndex + 5); // 5 is length of "data/"
    } else if (
      filePath.startsWith("recordings/") ||
      filePath.startsWith("audio/") ||
      filePath.startsWith("images/") ||
      filePath.startsWith("transcripts/")
    ) {
      // If it's already a relative path starting with the subdirectory, use it as is
      relativePath = filePath;
    } else if (filePath.includes("/recordings/")) {
      // Extract from after the last "/recordings/"
      const recordingsIndex = filePath.lastIndexOf("/recordings/");
      relativePath = "recordings/" + filePath.substring(recordingsIndex + 12);
    } else if (filePath.includes("/audio/")) {
      const audioIndex = filePath.lastIndexOf("/audio/");
      relativePath = "audio/" + filePath.substring(audioIndex + 7);
    } else if (filePath.includes("/images/")) {
      const imagesIndex = filePath.lastIndexOf("/images/");
      relativePath = "images/" + filePath.substring(imagesIndex + 8);
    } else if (filePath.includes("/transcripts/")) {
      const transcriptsIndex = filePath.lastIndexOf("/transcripts/");
      relativePath = "transcripts/" + filePath.substring(transcriptsIndex + 13);
    }

    // Create API URL
    return `${API_BASE_URL}/files/${relativePath}`;
  };

  // Helper function to convert video path to API URL (for backwards compatibility)
  const getVideoUrl = (videoPath) => {
    return getFileUrl(videoPath);
  };

  // Helper function to convert audio path to API URL
  const getAudioUrl = (audioPath) => {
    return getFileUrl(audioPath);
  };

  const handleStartRecording = async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

      const response = await fetch(`${API_BASE_URL}/camera/start`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          camera_index: 0,
          fps: 10.0,
          min_area: 2000,
          inactivity_timeout: 5.0,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Check if response is ok before trying to parse JSON
      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch (e) {
          errorData = {
            error: `Server error: ${response.status} ${response.statusText}`,
          };
        }
        console.error("Failed to start camera:", errorData);
        alert(
          `Failed to start camera: ${
            errorData.error || `Server error (${response.status})`
          }`
        );
        return;
      }

      const data = await response.json();

      if (data.error) {
        console.error("Failed to start camera:", data);
        alert(`Failed to start camera: ${data.error}`);
      } else {
        setIsRecording(true);
        console.log("Camera started:", data);
      }
    } catch (error) {
      console.error("Error starting camera:", error);

      if (error.name === "AbortError") {
        alert(
          "Connection timeout. Please make sure the backend server is running:\n\npython3 Backend/app.py"
        );
      } else if (
        error.message === "Load failed" ||
        error.message.includes("fetch") ||
        error.message.includes("NetworkError") ||
        error.message.includes("Failed to fetch") ||
        error instanceof TypeError
      ) {
        // Network/CORS error
        alert(
          "Could not connect to the backend server.\n\n" +
            "Please make sure the server is running:\n" +
            "python3 Backend/app.py\n\n" +
            "The server should be running on: http://localhost:5000"
        );
      } else {
        alert(`Error starting camera: ${error.message || "Unknown error"}`);
      }
    }
  };

  const handleStopRecording = async () => {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

      const response = await fetch(`${API_BASE_URL}/camera/stop`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Check if response is ok before trying to parse JSON
      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch (e) {
          errorData = {
            error: `Server error: ${response.status} ${response.statusText}`,
          };
        }
        console.error("Failed to stop camera:", errorData);
        alert(
          `Failed to stop camera: ${
            errorData.error || `Server error (${response.status})`
          }`
        );
        return;
      }

      const data = await response.json();

      if (data.error) {
        console.error("Failed to stop camera:", data);
        alert(`Failed to stop camera: ${data.error}`);
      } else {
        setIsRecording(false);
        setMotionDetected(false);
        setIsCurrentlyRecording(false);
        console.log("Camera stopped:", data);
      }
    } catch (error) {
      console.error("Error stopping camera:", error);

      if (error.name === "AbortError") {
        alert(
          "Connection timeout. Please make sure the backend server is running:\n\npython3 Backend/app.py"
        );
      } else if (
        error.message === "Load failed" ||
        error.message.includes("fetch") ||
        error.message.includes("NetworkError") ||
        error.message.includes("Failed to fetch") ||
        error instanceof TypeError
      ) {
        // Network/CORS error
        alert(
          "Could not connect to the backend server.\n\n" +
            "Please make sure the server is running:\n" +
            "python3 Backend/app.py\n\n" +
            "The server should be running on: http://localhost:5000"
        );
      } else {
        alert(`Error stopping camera: ${error.message || "Unknown error"}`);
      }
    }
  };

  // Fetch memory nodes on mount and periodically
  useEffect(() => {
    const fetchAndRefresh = async () => {
      await fetchMemoryNodes();
    };

    fetchAndRefresh();

    // Refresh events every 5 seconds to catch new recordings
    const refreshInterval = setInterval(fetchAndRefresh, 5000);

    return () => {
      clearInterval(refreshInterval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refresh events when recording stops
  useEffect(() => {
    if (!isRecording) {
      // Wait a bit for backend to finish processing, then refresh
      const timeoutId = setTimeout(() => {
        fetchMemoryNodes();
      }, 3000); // Wait 3 seconds after recording stops

      return () => clearTimeout(timeoutId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRecording]);

  // Update selectedEvent when events array updates (e.g., when summary is generated)
  useEffect(() => {
    if (selectedEvent) {
      // Find the updated event in the events array
      const updatedEvent = events.find((e) => e.id === selectedEvent.id);
      if (updatedEvent) {
        // Only update if the event data has actually changed
        const summaryChanged = updatedEvent.summary !== selectedEvent.summary;
        const transcriptChanged =
          updatedEvent.transcript !== selectedEvent.transcript;

        if (summaryChanged || transcriptChanged) {
          setSelectedEvent(updatedEvent);
        }
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events]);

  // Poll camera status regularly ONLY when recording is active
  useEffect(() => {
    let intervalId = null;

    // Only poll if recording is active
    if (!isRecording) {
      // Clear any existing interval
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
      return;
    }

    const checkCameraStatus = async () => {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000); // 2 second timeout

        const response = await fetch(`${API_BASE_URL}/camera/status`, {
          signal: controller.signal,
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        }).catch(() => null); // Catch network errors silently

        clearTimeout(timeoutId);

        // If fetch failed (network error), response will be null
        if (!response || !response.ok) {
          return;
        }

        try {
          const data = await response.json();

          // Update motion and recording states
          if (data.motion_detected !== undefined) {
            setMotionDetected(data.motion_detected);
          }
          if (data.is_currently_recording !== undefined) {
            setIsCurrentlyRecording(data.is_currently_recording);
          }

          // If camera stopped on backend, update state
          if (data.is_running === false) {
            setIsRecording(false);
            setMotionDetected(false);
            setIsCurrentlyRecording(false);
          }
        } catch (e) {
          // JSON parse error - silently ignore
        }
      } catch (error) {
        // Catch all errors silently
      }
    };

    // Start polling immediately when recording starts
    checkCameraStatus();
    // Poll every 1 second for responsive updates during recording
    intervalId = setInterval(checkCameraStatus, 1000);

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isRecording]);

  const handleRecordingStart = () => {
    console.log("Recording started");
  };

  const handleRecordingStop = (blob) => {
    console.log("Recording stopped", blob);
  };

  const generateAndPlayAnswer = async (
    query,
    summary,
    videoPath,
    audioPath
  ) => {
    if (!query || !summary) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/generate-answer-audio`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          summary: summary,
          video_path: videoPath || null,
          audio_path: audioPath || null,
        }),
      });

      if (!response.ok) {
        console.error("Failed to generate answer audio:", response.status);
        return;
      }

      // Check if response is audio or JSON
      const contentType = response.headers.get("content-type");

      if (contentType && contentType.includes("audio/mpeg")) {
        // Get the answer text from header if available
        const answerText = response.headers.get("X-Answer-Text");
        if (answerText) {
          console.log("Answer:", answerText);
        }

        // Get audio blob and play it
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);

        audio.play().catch((error) => {
          console.error("Error playing audio:", error);
        });

        // Clean up URL after playback
        audio.addEventListener("ended", () => {
          URL.revokeObjectURL(audioUrl);
        });
      } else {
        // If not audio, it might be JSON with error or answer text
        const data = await response.json();
        if (data.answer) {
          console.log("Answer:", data.answer);
          // Could use browser TTS as fallback if needed
          if ("speechSynthesis" in window) {
            const utterance = new SpeechSynthesisUtterance(data.answer);
            window.speechSynthesis.speak(utterance);
          }
        }
      }
    } catch (error) {
      console.error("Error generating answer audio:", error);
    }
  };

  const saveEventToJSON = async (event) => {
    // Extract the required fields: summary, transcript, title, timestamp, and file paths
    const eventData = {
      title: event.title || null,
      timestamp: event.timestamp || null,
      summary: event.summary || null,
      transcript: event.transcript || null,
      video_path: event.video_path || null,
      audio_path: event.audio_path || null,
      transcript_path: event.transcript_path || null,
      thumbnail_path: event.thumbnail_path || null,
    };

    try {
      // Send event data to backend to save to target.json
      const response = await fetch(`${API_BASE_URL}/save-event`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(eventData),
      });

      if (!response.ok) {
        console.error("Failed to save event data:", response.status);
      } else {
        console.log("Event data saved to target.json");
      }
    } catch (error) {
      console.error("Error saving event data:", error);
    }
  };

  const handleChatMessage = async (message) => {
    if (!message || !message.trim()) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/memory-nodes/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: message.trim(),
          max_results: 1, // Get only the most relevant event
        }),
      });

      if (!response.ok) {
        console.error("Failed to search memory nodes:", response.status);
        alert("Failed to search events. Please try again.");
        return;
      }

      const data = await response.json();
      const results = data.memory_nodes || [];

      if (results.length > 0) {
        // Get the most relevant memory node (first result)
        const mostRelevantNode = results[0];

        // Convert the memory node to an event format
        let metadata = {};
        try {
          metadata =
            typeof mostRelevantNode.metadata === "string"
              ? JSON.parse(mostRelevantNode.metadata)
              : mostRelevantNode.metadata || {};
        } catch (e) {
          console.error("Error parsing metadata:", e);
        }

        const summaryText = metadata.summary;
        const isSummaryLoading =
          summaryText === "Loading Summary..." ||
          summaryText === null ||
          !summaryText ||
          summaryText.trim() === "";
        const fullSummary = isSummaryLoading
          ? "Loading Summary..."
          : summaryText || "No summary available";

        let eventTitle = metadata.title;
        if (!eventTitle || eventTitle.trim() === "") {
          const timestamp =
            mostRelevantNode.timestamp || new Date().toISOString();
          let utcTimestamp = timestamp;
          if (
            !utcTimestamp.endsWith("Z") &&
            !utcTimestamp.includes("+") &&
            !utcTimestamp.includes("-", 10)
          ) {
            utcTimestamp = utcTimestamp.replace(/\.\d{3,6}/, "") + "Z";
          }

          const date = new Date(utcTimestamp);
          const formatter = new Intl.DateTimeFormat("en-US", {
            month: "2-digit",
            day: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
            hour12: true,
            timeZone: "America/New_York",
          });
          eventTitle = formatter.format(date);
        }

        const event = {
          id: mostRelevantNode.id,
          title: eventTitle,
          timestamp: mostRelevantNode.timestamp || new Date().toISOString(),
          summary: fullSummary,
          transcript: metadata.transcript || null,
          video_path: metadata.video_path || mostRelevantNode.file_path,
          audio_path: metadata.audio_path || null,
          transcript_path: metadata.transcript_path || null,
          thumbnail_path: metadata.thumbnail_path || null,
          objects_detected: metadata.objects_detected || [],
        };

        // Save event data to JSON file
        saveEventToJSON(event);

        // Open the popup with the most relevant event
        setSelectedEvent(event);

        // Generate answer and play audio
        generateAndPlayAnswer(
          message.trim(),
          fullSummary,
          event.video_path,
          event.audio_path
        );
      } else {
        // No results found
        alert("No relevant events found for your query.");
      }
    } catch (error) {
      console.error("Error searching memory nodes:", error);
      alert(
        "Failed to search events. Please make sure the backend server is running."
      );
    }
  };

  return (
    <main className="flex-1 bg-gradient-to-br from-primary-50 via-white to-primary-50 min-h-screen">
      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-3 gap-6 mb-6 items-stretch">
          <div className="col-span-2 flex flex-col">
            <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-6 flex flex-col h-full">
              <div className="mb-4 w-3/4 h-[25vh] flex items-center justify-center flex-shrink-0 mx-auto">
                <div className="w-full h-full">
                  <CameraRecorder
                    onRecordingStart={handleRecordingStart}
                    onRecordingStop={handleRecordingStop}
                    isRecording={isRecording}
                    motionDetected={motionDetected}
                    isCurrentlyRecording={isCurrentlyRecording}
                  />
                </div>
              </div>

              <div className="flex flex-row items-center justify-center gap-4 flex-shrink-0">
                <button
                  onClick={handleStartRecording}
                  disabled={isRecording}
                  className={`inline-flex items-center gap-2 px-8 py-3 rounded-xl font-semibold transition-all duration-200 shadow-lg ${
                    isRecording
                      ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                      : "bg-primary-600 text-white hover:bg-primary-700 hover:shadow-xl transform hover:-translate-y-0.5"
                  }`}
                >
                  Start Recording
                </button>
                <button
                  onClick={handleStopRecording}
                  disabled={!isRecording}
                  className={`inline-flex items-center gap-2 px-8 py-3 rounded-xl font-semibold transition-all duration-200 shadow-lg ${
                    !isRecording
                      ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                      : "bg-red-600 text-white hover:bg-red-700 hover:shadow-xl transform hover:-translate-y-0.5"
                  }`}
                >
                  Stop Recording
                </button>
              </div>
            </div>
          </div>

          <div className="col-span-1 flex flex-col">
            <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-6 flex flex-col h-full">
              <div className="mb-3 flex-shrink-0">
                <h2 className="text-xl font-bold text-primary-900 mb-1">
                  Event Timeline
                </h2>
                <p className="text-sm text-gray-600">
                  {events.length} {events.length === 1 ? "event" : "events"}{" "}
                  recorded
                </p>
              </div>

              <div className="flex-1 min-h-0" style={{ height: "25vh" }}>
                <Timeline events={events} onEventClick={handleEventClick} />
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 p-6">
          <div className="mb-4">
            <h2 className="text-xl font-bold text-primary-900 mb-1">
              AI Assistant
            </h2>
            <p className="text-sm text-gray-600">
              Ask questions about recorded events
            </p>
          </div>
          <Chat onSendMessage={handleChatMessage} />
        </div>
      </div>

      {selectedEvent && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
          onClick={closeEventModal}
        >
          <div
            className="bg-white rounded-2xl shadow-2xl max-w-6xl w-full max-h-[80vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between rounded-t-2xl">
              <div>
                <h2 className="text-2xl font-bold text-primary-900">
                  {selectedEvent.title}
                </h2>
                {selectedEvent.timestamp && (
                  <p className="text-sm text-gray-500 mt-1">
                    {(() => {
                      // Ensure timestamp is treated as UTC
                      let utcTimestamp = selectedEvent.timestamp;
                      if (
                        !utcTimestamp.endsWith("Z") &&
                        !utcTimestamp.includes("+") &&
                        !utcTimestamp.includes("-", 10)
                      ) {
                        utcTimestamp =
                          utcTimestamp.replace(/\.\d{3,6}/, "") + "Z";
                      }

                      // Parse as UTC and format in EST/EDT
                      const date = new Date(utcTimestamp);
                      const formatter = new Intl.DateTimeFormat("en-US", {
                        month: "2-digit",
                        day: "2-digit",
                        year: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                        hour12: true,
                        timeZone: "America/New_York",
                      });
                      return formatter.format(date);
                    })()}
                  </p>
                )}
              </div>
              <button
                onClick={closeEventModal}
                className="text-gray-400 hover:text-gray-600 transition-colors p-2 hover:bg-gray-100 rounded-lg"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-6 w-6"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
            <div className="p-6">
              <div className="flex gap-6">
                {/* Left Column: Video and Audio */}
                {(selectedEvent.video_path || selectedEvent.audio_path) && (
                  <div className="flex-shrink-0 w-1/2 flex flex-col gap-6">
                    {selectedEvent.video_path && (
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900 mb-3">
                          Video Recording
                        </h3>
                        <div className="rounded-lg overflow-hidden bg-black sticky top-6">
                          <video
                            controls
                            className="w-full h-auto"
                            src={getVideoUrl(selectedEvent.video_path)}
                          >
                            Your browser does not support the video tag.
                          </video>
                        </div>
                      </div>
                    )}

                    {selectedEvent.audio_path && (
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900 mb-3">
                          Audio Recording
                        </h3>
                        <div className="rounded-lg overflow-hidden bg-gray-100 p-4">
                          <audio
                            controls
                            className="w-full"
                            src={getAudioUrl(selectedEvent.audio_path)}
                          >
                            Your browser does not support the audio tag.
                          </audio>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Right Column: Summary, Transcript, Objects */}
                <div
                  className={`flex-1 ${
                    selectedEvent.video_path || selectedEvent.audio_path
                      ? ""
                      : "w-full"
                  }`}
                >
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">
                    Event Summary
                  </h3>
                  <div
                    className="bg-black/50 rounded-lg p-2 overflow-y-auto"
                    style={{ maxHeight: "33vh" }}
                  >
                    {selectedEvent.summary === "Loading Summary..." ? (
                      <div className="flex items-center gap-2">
                        <svg
                          className="animate-spin h-5 w-5 text-black"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                          ></circle>
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          ></path>
                        </svg>
                        <p className="text-black italic">Loading Summary...</p>
                      </div>
                    ) : (
                      <div className="text-black leading-relaxed break-words whitespace-pre-line">
                        {formatSummary(
                          selectedEvent.summary ||
                            "No summary available for this event."
                        )}
                      </div>
                    )}
                  </div>

                  {selectedEvent.transcript && (
                    <div className="mt-6">
                      <h4 className="text-md font-semibold text-gray-900 mb-2">
                        Transcript
                      </h4>
                      <div className="bg-gray-50 rounded-lg p-4">
                        <p className="text-sm text-gray-700 whitespace-pre-wrap">
                          {selectedEvent.transcript}
                        </p>
                      </div>
                    </div>
                  )}

                  {selectedEvent.objects_detected &&
                    selectedEvent.objects_detected.length > 0 && (
                      <div className="mt-6">
                        <h4 className="text-md font-semibold text-gray-900 mb-2">
                          Objects Detected
                        </h4>
                        <div className="flex flex-wrap gap-2">
                          {selectedEvent.objects_detected.map((obj, idx) => (
                            <span
                              key={idx}
                              className="px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm font-medium"
                            >
                              {obj}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

export default Recording;
