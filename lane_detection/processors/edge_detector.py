"""Edge detection module for lane detection."""

import cv2
import numpy as np


class EdgeDetector:
    """Class for detecting edges in an image using various algorithms."""

    def __init__(self, low_threshold=50, high_threshold=150, kernel_size=5):
        """
        Initialize the edge detector with the given parameters.

        Args:
            low_threshold (int, optional): Low threshold for Canny edge detection.
                Defaults to 50.
            high_threshold (int, optional): High threshold for Canny edge detection.
                Defaults to 150.
            kernel_size (int, optional): Kernel size for Gaussian blur. Must be odd.
                Defaults to 5.
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.kernel_size = kernel_size

    def detect_edges(self, image):
        """
        Detect edges in the given image using Canny edge detection.

        Args:
            image (numpy.ndarray): Input image to detect edges in.

        Returns:
            numpy.ndarray: Binary image with detected edges.
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.kernel_size, self.kernel_size), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)

        return edges

    def sobel_edge_detection(self, image, orient='x', sobel_kernel=3, thresh=(20, 100)):
        """
        Apply Sobel edge detection to highlight gradients in a specific orientation.

        Args:
            image (numpy.ndarray): Input image.
            orient (str, optional): Orientation of gradient to detect ('x' or 'y'). 
                Defaults to 'x'.
            sobel_kernel (int, optional): Size of the Sobel kernel. Must be odd.
                Defaults to 3.
            thresh (tuple, optional): Thresholds for binary output. 
                Defaults to (20, 100).

        Returns:
            numpy.ndarray: Binary image with detected edges.
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Sobel in x or y direction
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Take absolute value and convert to uint8
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # Create binary mask
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return binary_output

    def magnitude_gradient(self, image, sobel_kernel=3, thresh=(30, 100)):
        """
        Apply magnitude gradient to detect edges based on combined x and y gradients.

        Args:
            image (numpy.ndarray): Input image.
            sobel_kernel (int, optional): Size of the Sobel kernel. Must be odd.
                Defaults to 3.
            thresh (tuple, optional): Thresholds for binary output. 
                Defaults to (30, 100).

        Returns:
            numpy.ndarray: Binary image with detected edges.
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Sobel in both x and y directions
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Convert to 8-bit scale
        scale_factor = np.max(gradient_magnitude) / 255
        gradient_magnitude = (gradient_magnitude / scale_factor).astype(np.uint8)
        
        # Create binary mask
        binary_output = np.zeros_like(gradient_magnitude)
        binary_output[(gradient_magnitude >= thresh[0]) & (gradient_magnitude <= thresh[1])] = 1
        
        return binary_output

    def direction_gradient(self, image, sobel_kernel=3, thresh=(0.7, 1.3)):
        """
        Apply direction gradient to detect edges based on gradient direction.

        Args:
            image (numpy.ndarray): Input image.
            sobel_kernel (int, optional): Size of the Sobel kernel. Must be odd.
                Defaults to 3.
            thresh (tuple, optional): Thresholds for binary output in radians. 
                Defaults to (0.7, 1.3).

        Returns:
            numpy.ndarray: Binary image with detected edges.
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2 and image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Sobel in both x and y directions
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Calculate gradient direction
        direction = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        
        # Create binary mask
        binary_output = np.zeros_like(direction)
        binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
        
        return binary_output

    def combined_thresholds(self, image):
        """
        Apply combined threshold techniques for comprehensive edge detection.

        Args:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Binary image with detected edges.
        """
        # Apply different threshold methods
        sobelx = self.sobel_edge_detection(image, orient='x', thresh=(20, 100))
        sobely = self.sobel_edge_detection(image, orient='y', thresh=(20, 100))
        mag_binary = self.magnitude_gradient(image, thresh=(30, 100))
        dir_binary = self.direction_gradient(image, thresh=(0.7, 1.3))
        
        # Combine thresholds
        combined = np.zeros_like(sobelx)
        combined[((sobelx == 1) & (sobely == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        
        return combined
