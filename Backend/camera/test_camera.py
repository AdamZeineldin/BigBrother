import cv2
import sys
import argparse
import time

def main():
    """
    A bare-minimum script to test opening a camera with OpenCV.
    """
    parser = argparse.ArgumentParser(description="Bare-minimum camera test.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="The index of the camera to test.",
    )
    args = parser.parse_args()

    camera_index = args.camera_index
    print(f"Attempting to open camera at index: {camera_index}")

    # Use CAP_AVFOUNDATION for better macOS compatibility
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}.")
        print("Please check camera permissions and that the index is correct.")
        return 1

    print("Camera opened successfully. Priming stream...")

    # --- Priming Loop ---
    # Some cameras (especially virtual/Continuity) need a moment to start streaming.
    # We'll try to read a frame for a few seconds before giving up.
    primed = False
    for i in range(30): # Try for ~3 seconds
        ret, _ = cap.read()
        if ret:
            print(f"Stream is live after {i + 1} attempts.")
            primed = True
            break
        time.sleep(0.1)

    if not primed:
        print("Error: Camera opened but failed to start streaming. Please try another camera index.")
        cap.release()
        return 1


    print("Priming successful. A window should appear. Press 'q' in the window to quit.")
    window_name = f"Camera Test (Index: {camera_index})"

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # If the frame was not read successfully, print an error and break the loop
        if not ret:
            print("Error: Failed to grab frame. The camera may have been disconnected.")
            break

        # Display the resulting frame
        cv2.imshow(window_name, frame)

        # Wait for 1ms and check if the 'q' key was pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed. Exiting.")
            break

    # When everything is done, release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
