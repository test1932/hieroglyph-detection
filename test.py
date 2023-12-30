import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

# Create a Shapely MultiPolygon
multipolygon = MultiPolygon([
    Polygon([(50, 50), (150, 50), (150, 150), (50, 150)]),
    Polygon([(100, 100), (200, 100), (200, 200), (100, 200)])
])

image_size = (300, 300, 3)
image = np.zeros(image_size, dtype=np.uint8)

# Iterate through each polygon in the MultiPolygon
for polygon in multipolygon.geoms:
    # Convert Shapely polygon to NumPy array
    polygon_np = np.array(polygon.exterior.coords, dtype=np.int32)

    # Reshape the array for OpenCV
    polygon_np = polygon_np.reshape((-1, 1, 2))

    # Draw the polygon on the image
    cv2.polylines(image, [polygon_np], isClosed=True, color=(255, 255, 255), thickness=2)

# Display the image
cv2.imshow('Image with Shapely MultiPolygon', image)
cv2.waitKey(0)
cv2.destroyAllWindows()