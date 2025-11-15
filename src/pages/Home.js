import React from "react";
import { Link } from "react-router-dom";

function Home() {
  return (
    <main className="flex-1 bg-background flex items-center justify-center flex-col">
      <h1 className="text-6xl font-bold text-text mb-2 w-fit">BigBrother</h1>
      <h1 className="text-2xl font-bold text-text mb-8 w-fit">
        Assisting alzheimer's patients with everyday tasks
      </h1>
      <Link
        to="/recording"
        className="bg-primary-600 text-white px-8 py-3 rounded-lg font-medium hover:bg-primary-700 transition-colors duration-200 shadow-sm"
      >
        Click here to get started
      </Link>
      <div className="container mx-auto px-6 py-8"></div>
    </main>
  );
}

export default Home;
