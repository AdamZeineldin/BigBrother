import React, { useState, useRef, useEffect } from "react";

function Timeline({ events = [] }) {
  const [hoveredIndex, setHoveredIndex] = useState(null);
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
      <div className="max-h-[50vh] flex flex-col items-center justify-center">
        <div className="text-gray-400 text-sm">No events yet</div>
      </div>
    );
  }

  const getEventTitle = (event) => {
    if (typeof event.title === "string") return event.title;
    if (typeof event.title === "object") return String(event.title);
    return event.label || "Event";
  };

  return (
    <div
      ref={scrollContainerRef}
      className="overflow-y-auto flex items-center justify-center relative"
      style={{ maxHeight: "50vh" }}
    >
      <div
        ref={contentRef}
        className="relative flex flex-col items-center py-8 gap-12 w-full"
      >
        <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-gray-300 transform -translate-x-1/2"></div>
        {events.map((event, idx) => {
          const title = getEventTitle(event);
          const isHovered = hoveredIndex === idx;
          const tooltipOnRight = idx % 2 === 0;

          return (
            <div
              key={event.id || idx}
              className={`relative flex items-center ${
                isHovered ? "cursor-pointer" : ""
              }`}
              onMouseEnter={() => setHoveredIndex(idx)}
              onMouseLeave={() => setHoveredIndex(null)}
            >
              <div className="relative z-10 w-4 h-4 rounded-full bg-gray-600 hover:bg-gray-800 cursor-pointer transition-colors"></div>

              <div
                className={`absolute z-20 ${
                  tooltipOnRight ? "left-6" : "right-6"
                } top-1/2 transform -translate-y-1/2 bg-white border border-gray-200 rounded-lg shadow-lg px-3 py-2 whitespace-nowrap`}
              >
                <div className="text-sm font-medium text-gray-800">{title}</div>
                <div
                  className={`absolute top-1/2 transform -translate-y-1/2 ${
                    tooltipOnRight ? "-left-1" : "-right-1"
                  } w-2 h-2 bg-white border-r border-b border-gray-200 rotate-45`}
                ></div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default Timeline;
