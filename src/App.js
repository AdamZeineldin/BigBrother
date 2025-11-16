import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Header from "./components/Header";
import Footer from "./components/Footer";
import Home from "./pages/Home";
import About from "./pages/About";
import FAQ from "./pages/FAQ";
import Recording from "./pages/Recording";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-primary-50 flex flex-col">
        <Header />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
          <Route path="/faq" element={<FAQ />} />
          <Route path="/recording" element={<Recording />} />
        </Routes>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
