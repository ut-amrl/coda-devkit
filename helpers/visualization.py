import pdb

#Libraries

#ROS
import std_msgs.msg
from sensor_msgs.msg import PointCloud2

#Custom
from helpers.constants import PointField

def pub_pc_to_rviz(pc, pc_pub, ts, frame_id="os_sensor"):
    is_intensity = pc.shape[-1]==4
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
    pc_msg.row_step     = pc_msg.width * pc.itemsize

    pc_msg.header            = std_msgs.msg.Header()
    pc_msg.header.stamp      = ts
    pc_msg.header.frame_id   = frame_id

    fields  = [
        PointField('x', 0, DATATYPE, 1),
        PointField('y', pc.itemsize, DATATYPE, 1),
        PointField('z', pc.itemsize*2, DATATYPE, 1)
    ]
    if is_intensity:
        pc_msg.point_step   = pc.itemsize*4
        fields.append(PointField('intensity', pc.itemsize*3, DATATYPE, 1))
    else:
        pc_msg.point_step   = pc.itemsize*3
    
    pc_msg.fields = fields
    pc_msg.data   = pc_flat.tobytes()
    # pdb.set_trace()
    pc_pub.publish(pc_msg)

