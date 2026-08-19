API Reference
=============

.. module:: cuvslam

This module provides Python bindings for the cuVSLAM library.

Main classes
------------

.. autoclass:: Odometry
   :members:

.. autoclass:: Slam
   :members:

.. autoclass:: Tracker
   :members:

Data Structures
---------------

.. autoclass:: Pose
   :members:

.. autoclass:: Distortion
   :members:

.. autoclass:: Camera
   :members:

.. autoclass:: ImuCalibration
   :members:

.. autoclass:: Rig
   :members:

.. autoclass:: PoseStamped
   :members:

.. autoclass:: PoseWithCovariance
   :members:

.. autoclass:: PoseEstimate
   :members:

.. autoclass:: ImuMeasurement
   :members:

.. autoclass:: Landmark
   :members:

.. autoclass:: Observation
   :members:

Functions
---------

.. autofunction:: get_version

.. autofunction:: set_verbosity

.. autofunction:: warm_up_gpu
