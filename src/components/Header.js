import React, { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  signInWithPopup,
  signOut,
  GoogleAuthProvider,
  getRedirectResult,
} from "firebase/auth";
import { auth } from "../firebase";

function Header() {
  const [user, setUser] = useState(null);
  const location = useLocation();

  useEffect(() => {
    getRedirectResult(auth)
      .then((result) => {
        if (result) {
          setUser(result.user);
        }
      })
      .catch((error) => {
        console.error(error);
      });

    const unsubscribe = auth.onAuthStateChanged((user) => {
      setUser(user);
    });

    return () => unsubscribe();
  }, []);

  const signInWithGoogle = async () => {
    const provider = new GoogleAuthProvider();
    provider.addScope("profile");
    provider.addScope("email");

    try {
      await signInWithPopup(auth, provider);
    } catch (error) {
      console.log(error);
    }
  };

  const handleSignOut = async () => {
    try {
      await signOut(auth);
    } catch (error) {
      console.error("Error signing out:", error);
    }
  };

  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <header className="bg-gradient-to-r from-primary-600 to-primary-600 text-white shadow-lg">
      <div className="relative px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex-shrink-0 w-1/3"></div>
          <nav className="flex items-center space-x-12 absolute left-1/2 transform -translate-x-1/2">
            <Link
              to="/"
              className={`font-medium transition-colors duration-200 ${
                isActive("/")
                  ? "text-white border-b-2 border-white pb-1"
                  : "text-primary-100 hover:text-white"
              }`}
            >
              Home
            </Link>
            <Link
              to="/about"
              className={`font-medium transition-colors duration-200 ${
                isActive("/about")
                  ? "text-white border-b-2 border-white pb-1"
                  : "text-primary-100 hover:text-white"
              }`}
            >
              About
            </Link>
            <Link
              to="/faq"
              className={`font-medium transition-colors duration-200 ${
                isActive("/faq")
                  ? "text-white border-b-2 border-white pb-1"
                  : "text-primary-100 hover:text-white"
              }`}
            >
              FAQ
            </Link>
          </nav>
          <div className="flex items-center space-x-4 flex-shrink-0 w-1/3 justify-end">
            {user ? (
              <div className="flex items-center space-x-3">
                <span className="hidden md:block font-medium text-white whitespace-nowrap">
                  Hi, {user.displayName || user.email}
                </span>
                <button
                  onClick={handleSignOut}
                  className="bg-white text-primary-700 px-4 py-2 rounded-lg font-medium hover:bg-primary-50 transition-colors duration-200 shadow-sm whitespace-nowrap"
                >
                  Sign Out
                </button>
              </div>
            ) : (
              <button
                onClick={signInWithGoogle}
                className="bg-white text-primary-700 px-6 py-2 rounded-lg font-medium hover:bg-primary-50 transition-colors duration-200 flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm whitespace-nowrap"
              >
                <span>Sign in</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;
