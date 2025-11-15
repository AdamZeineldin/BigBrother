import React from "react";
import { Link } from "react-router-dom";

function Home() {
  return (
    <main className="flex-1 bg-background">
      <div className="container mx-auto px-6 py-12">
        <div className="text-center mb-16">
          <h1 className="text-6xl font-bold text-primary-800 mb-4">
            BigBrother
          </h1>
          <h2 className="text-2xl font-semibold text-gray-700 mb-8 max-w-2xl mx-auto">
            Assisting Alzheimer's patients with everyday tasks
          </h2>
        </div>

        <div className="rounded-lg p-8 mb-16 bg-white rounded-lg p-8 shadow-md border border-gray-100">
          <h3 className="text-3xl font-bold text-primary-800 mb-6 text-center">
            How It Works
          </h3>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-primary-600 text-white w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold mx-auto mb-4">
                1
              </div>
              <h4 className="text-xl font-semibold text-primary-800 mb-2">
                Start Recording
              </h4>
              <p className="text-gray-700">
                Begin by recording your surroundings and activities.
              </p>
            </div>
            <div className="text-center">
              <div className="bg-primary-600 text-white w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold mx-auto mb-4">
                2
              </div>
              <h4 className="text-xl font-semibold text-primary-800 mb-2">
                Track Events
              </h4>
              <p className="text-gray-700">
                Events are detected and logged on a custom timeline
                automatically utilizing computer vision and AI analysis.
              </p>
            </div>
            <div className="text-center">
              <div className="bg-primary-600 text-white w-12 h-12 rounded-full flex items-center justify-center text-xl font-bold mx-auto mb-4">
                3
              </div>
              <h4 className="text-xl font-semibold text-primary-800 mb-2">
                Query assistance
              </h4>
              <p className="text-gray-700">
                Use the chat box to ask various questions about past events by
                typing or using speech to text.
              </p>
            </div>
          </div>
        </div>

        <div className="text-center p-8">
          <h3 className="text-2xl font-bold text-primary-800 mb-4">
            Ready to try now?
          </h3>
          <Link
            to="/recording"
            className="inline-block bg-primary-600 text-white px-8 py-3 rounded-lg font-medium hover:bg-primary-700 transition-colors duration-200 shadow-sm"
          >
            Get Started →
          </Link>
        </div>
      </div>
    </main>
  );
}

export default Home;
