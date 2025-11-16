import React, { useRef, useEffect } from "react";

function Timeline({ events = [], onEventClick }) {
  const scrollContainerRef = useRef(null);
  const contentRef = useRef(null);

  useEffect(() => {
    if (scrollContainerRef.current && contentRef.current && events.length > 0) {
      scrollContainerRef.current.scrollTop =
        scrollContainerRef.current.scrollHeight;
    }
  }, [events.length]);

  if (events.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center py-8">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-12 w-12 text-gray-300 mb-3"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 8v4l3 3m-6 0V8m6 0l-3 3m0 0l3 3m-3-3H9"
          />
        </svg>
        <div className="text-gray-400 text-sm font-medium">No events yet</div>
        <div className="text-gray-400 text-xs mt-1">
          Events will appear here
        </div>
      </div>
    );
  }

  const getEventTitle = (event) => {
    if (typeof event.title === "string") return event.title;
    if (typeof event.title === "object") return String(event.title);
    return event.label || "Event";
  };

  const handleEventClick = (event) => {
    if (onEventClick) {
      onEventClick(event);
    }
  };

  return (
    <div ref={scrollContainerRef} className="overflow-y-auto relative h-full">
      <div
        ref={contentRef}
        className="relative flex flex-col items-start pt-2 pb-2 gap-2 w-full"
      >
        <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gray-300"></div>
        {events.map((event, idx) => {
          const title = getEventTitle(event);

          return (
            <div
              key={event.id || idx}
              className="relative flex items-center w-full group"
            >
              <div className="absolute left-6 top-1/2 transform -translate-x-1/2 -translate-y-1/2 z-10 w-4 h-4 rounded-full bg-primary-600 hover:bg-primary-700 transition-all duration-200 shadow-md hover:shadow-lg border-2 border-white"></div>

              <div className="ml-10 flex-1 min-w-0">
                <button
                  onClick={() => handleEventClick(event)}
                  className="text-left bg-white border border-gray-200 rounded-lg shadow-sm px-2.5 py-1.5 hover:shadow-md hover:border-primary-300 transition-all duration-200 cursor-pointer w-full group-hover:bg-primary-50"
                >
                  <div className="text-xs font-semibold text-gray-900 group-hover:text-primary-700 truncate">
                    {title}
                  </div>
                  {event.timestamp && (
                    <div className="text-xs text-gray-500 mt-0.5">
                      {(() => {
                        // Ensure timestamp is treated as UTC - add 'Z' if not present
                        let utcTimestamp = event.timestamp;
                        if (
                          !utcTimestamp.endsWith("Z") &&
                          !utcTimestamp.includes("+") &&
                          !utcTimestamp.includes("-", 10)
                        ) {
                          // If no timezone info, assume it's UTC and add 'Z'
                          utcTimestamp = utcTimestamp.replace(/\.\d{3,6}/, "") + "Z";
                        }

                        // Parse as UTC and format in EST/EDT
                        const date = new Date(utcTimestamp);
                        const formatter = new Intl.DateTimeFormat("en-US", {
                          month: "2-digit",
                          day: "2-digit",
                          hour: "2-digit",
                          minute: "2-digit",
                          hour12: true,
                          timeZone: "America/New_York",
                        });
                        return formatter.format(date);
                      })()}
                    </div>
                  )}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default Timeline;
