### Illumination

While I will give a full description of the current illumination system, I higly recoment considering a cheaper and more compact alternative:

https://lumencor.com/wp-content/uploads/sites/11/2018/10/LUM.54-10043.CELESTA.pdf

![](https://github.com/BogdanBintu/ChromatinImagingV2/blob/master/MicroscopeHelp/Illumination/Illumination_simplified_scheme.PNG)

The system has 5 laser lines:

![](https://github.com/BogdanBintu/ChromatinImagingV2/blob/master/MicroscopeHelp/Illumination/laser_heads.jpg)

High power:
* 500mW 750nm laser from MPB
* 1.5W 647nm laser from MPB
* 1.5W 561nm laser from MPB
* 488nm laser from Coherent
Low power:
* 405nm laser from OBIS

After each laser we recommend a 3-lens zoom system allowing for full control over the width of the laser.
ND filters can also be used to reduce the power if necessary.
All lasers are controled to a 2mm width as required by the AOTF. 
The beam size can be measured using a beam profiler (https://www.newport.com/f/laser-beam-profilers).

![](https://github.com/BogdanBintu/ChromatinImagingV2/blob/master/MicroscopeHelp/Illumination/1-LaserZoomSystem.png)

The beam of each laser is controlled by a pair of 2 mirrors and combined into the dichroic cage.
The combined beam of 488, 561, 647 and 750 lines feeds into the acusto-optical tunable filter (AOTF).
The frequency of the AOTF controls the illumination power of each of the lines.
The small OBIS laser has its own separate control.
![](https://github.com/BogdanBintu/ChromatinImagingV2/blob/master/MicroscopeHelp/Illumination/2-LaserMirrorsAndLaserDichroics.png)

In the initial instance the beam was expanded to ~5cm untill the beam filled the camera.
The expansion was done via a 3 lens zoom system as indicated below.
This produces a gaussian profile on the camera and is generally good for small fields of view.
Moreover such a telescope is necessary for instaling the Lumencor Celesta system or a fiber-coupled laser beam.
![](https://github.com/BogdanBintu/ChromatinImagingV2/blob/master/MicroscopeHelp/Illumination/3-MasterZoomSystem.png)

To use the entire filed of the camera I installed a beam shaper. The beamshaper is very sensitive on the input width.
Thus a 3lens-zoom is used before and after the beamshaper.

![](https://github.com/BogdanBintu/ChromatinImagingV2/blob/master/MicroscopeHelp/Illumination/PiShaperBeam.png)

