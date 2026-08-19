API Reference
=============

.. module:: cuvslam

This module provides Python bindings for the cuVSLAM library.

``Odometry`` and ``Slam`` are available directly in this module. The former
``cuvslam.core`` namespace remains as a deprecated compatibility alias.

Core classes
------------

.. autoclass:: Odometry
   :members:

.. autoclass:: Slam
   :members:

.. autoclass:: cuvslam.Odometry.Config
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

.. autoclass:: PoseEstimate
   :members:

.. autoclass:: ImuMeasurement
   :members:

.. autoclass:: Landmark
   :members:

.. autoclass:: Observation
   :members:

Tracker class
-------------

``Tracker`` coordinates calls that advance odometry and SLAM. Use
``get_odometry()`` and ``get_slam()`` for module-specific queries and operations;
do not call either returned component's ``track()`` method.

.. autoclass:: Tracker
   :members:
   :undoc-members:

Functions
---------

.. autofunction:: get_version

.. autofunction:: set_verbosity

.. autofunction:: warm_up_gpu
