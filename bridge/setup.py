from setuptools import find_packages, setup

package_name = 'bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='airou',
    maintainer_email='arikak@ou.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ackermann_bridge = bridge.ackermann_bridge:main',
            'ros_to_tcp_bridge = bridge.ros_to_tcp_bridge:main',
            'real_robot_bridge = bridge.real_robot_bridge:main',
        ],
    },
)
