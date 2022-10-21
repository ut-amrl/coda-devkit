import pdb

#Libraries
import numpy as np

#ROS
import std_msgs.msg
from sensor_msgs.msg import PointCloud2

#Custom
from helpers.constants import PointField

def pub_pc_to_rviz(pc, pc_pub, ts, frame_id="os_sensor"):
    is_intensity    = pc.shape[-1]>=4
    is_ring         = pc.shape[-1]>=5
    if is_ring:
        ring        = pc[:, :, 4].astype(np.uint16)
        pc_noring   = pc[:, :, :4]

        pc_bytes        = pc_noring.reshape(-1, 1).tobytes()
        ring_bytes      = ring.reshape(-1, 1).tobytes()

        pc_bytes_np = np.frombuffer(pc_bytes, dtype=np.uint8)
        ring_bytes_np = np.frombuffer(ring_bytes, dtype=np.uint8).reshape(-1, 2)

        pc_bytes_np = pc_bytes_np.reshape(-1, 16)
        all_bytes_np= np.hstack((pc_bytes_np, ring_bytes_np)).reshape(-1, 1)

    pc_flat = pc.reshape(-1, 1)

    DATATYPE= PointField.FLOAT32
    if pc.itemsize==PointField.UINT16:
        DATATYPE = PointField.FLOAT32
    else:
        DATATYPE = PointField.FLOAT64
        print("Undefined datatype size accessed, defaulting to FLOAT64...")

    pc_msg = PointCloud2()
    pc_msg.width        = pc.shape[1]
    pc_msg.height       = pc.shape[0]

    pc_msg.header            = std_msgs.msg.Header()
    pc_msg.header.stamp      = ts
    pc_msg.header.frame_id   = frame_id

    fields  = [
        PointField('x', 0, DATATYPE, 1),
        PointField('y', pc.itemsize, DATATYPE, 1),
        PointField('z', pc.itemsize*2, DATATYPE, 1)
    ]

    if is_ring:
        pc_msg.point_step   = pc.itemsize*4 + 2
        fields.append(PointField('intensity', pc.itemsize*3, DATATYPE, 1))
        fields.append(PointField('ring', pc.itemsize*4, PointField.UINT16, 1))
    elif is_intensity:
        pc_msg.point_step   = pc.itemsize*4
        fields.append(PointField('intensity', pc.itemsize*3, DATATYPE, 1))
    else:
        pc_msg.point_step   = pc.itemsize*3
    pc_msg.row_step     = pc_msg.width * pc_msg.point_step

    pc_msg.fields   = fields
    if is_ring:
        pc_msg.data     = all_bytes_np.tobytes()
    else:
        pc_msg.data     = pc_flat.tobytes()
    pc_msg.is_dense = True
    pc_pub.publish(pc_msg)

    return pc_msg

