import folium
from geopy import distance
from IPython.display import Image, display
import numpy as np

def convert_xyz_to_latlon(x, y, z):
    # Define the origin coordinates (latitude, longitude) and altitude
    origin = (30.28805556, -97.7375, 0)  # Ut Austin Start Coordinates
    origin = (30.288114, -97.737699, 0)

    trans = np.array([[-1, 0], [0, -1]])
    trans_xy = trans @ np.array([x, y]).reshape(2, 1)

    # Calculate the destination coordinates based on the XYZ offsets
    # dest = distance.distance(meters=z).destination(origin[:2], 90 - y / 111111, x / 111111)
    # import pdb; pdb.set_trace()
    offset = np.linalg.norm(trans_xy)
    bearing =  np.arctan2(trans_xy[1].item(), trans_xy[0].item())

    dest = distance.distance(meters=offset).destination(origin[:2], bearing=90 - bearing * 180 / np.pi)

    # Return the latitude, longitude, and altitude
    return dest[0], dest[1], origin[2] + z

def add_marker(x, y, z, m):
    latitude, longitude, altitude = convert_xyz_to_latlon(x, y, z)
    # folium.Marker(location=[latitude, longitude]).add_to(m)
    folium.CircleMarker(location=[latitude, longitude],
                        radius=2, weight=1).add_to(m)
    return m

def main():
    trajectory_list = [0, 1, 2, 6]
    # import pdb; pdb.set_trace()
    latitude, longitude, altitude = convert_xyz_to_latlon(0, 0, 0)
    m = folium.Map(location=[latitude, longitude], zoom_start=35)
    for trajectory in trajectory_list:
        pose_path = "/home/arthur/Downloads/%s.txt"%str(trajectory)
        pose_np = np.loadtxt(pose_path).reshape(-1, 8)

        for pose in pose_np:
            _, x, y, z, _, _, _, _ = pose
            print(x, y, z)
            m = add_marker(x, y, z, m)

    map_filename = "./map_image.html"
    m.fit_bounds(m.get_bounds())
    m.save(map_filename)

    print("Map saved as HTML:", map_filename)

    # # XYZ coordinates
    # x = 0
    # y = 0
    # z = 0

    # # Convert XYZ to latitude, longitude, and altitude
    # latitude, longitude, altitude = convert_xyz_to_latlon(x, y, z)
    # print("Latitude:", latitude)
    # print("Longitude:", longitude)
    # print("Altitude:", altitude)

    # # Create a map centered at the converted coordinates
    # m = folium.Map(location=[latitude, longitude], zoom_start=35)

    # # Add a marker at the converted coordinates
    # folium.Marker(location=[latitude, longitude]).add_to(m)

    # # XYZ coordinates
    # x = 0
    # y = 5
    # z = 0

    # # Convert XYZ to latitude, longitude, and altitude
    # latitude, longitude, altitude = convert_xyz_to_latlon(x, y, z)
    # # Add a marker at the converted coordinates
    # folium.Marker(location=[latitude, longitude]).add_to(m)


if __name__ == '__main__':
    main()

# XYZ coordinates
# x = 0
# y = 0
# z = 0

# # Convert XYZ to latitude, longitude, and altitude
# latitude, longitude, altitude = convert_xyz_to_latlon(x, y, z)
# print("Latitude:", latitude)
# print("Longitude:", longitude)
# print("Altitude:", altitude)

# # Create a map centered at the converted coordinates
# m = folium.Map(location=[latitude, longitude], zoom_start=35)

# # Add a marker at the converted coordinates
# folium.Marker(location=[latitude, longitude]).add_to(m)

# # XYZ coordinates
# x = 0
# y = 5
# z = 0

# # Convert XYZ to latitude, longitude, and altitude
# latitude, longitude, altitude = convert_xyz_to_latlon(x, y, z)
# # Add a marker at the converted coordinates
# folium.Marker(location=[latitude, longitude]).add_to(m)

# map_filename = "./map_image.html"
# m.save(map_filename)

# print("Map saved as HTML:", map_filename)