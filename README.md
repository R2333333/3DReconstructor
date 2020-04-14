# 3DReconstructor

## Construction of 3D Visual Representations From 2D Images


Authors:
---
Roy Xu, Jiangyuan Wu, Wayne Du, Zhaohong Wan


Description:
---
With regular cameras, capturing 2D images of a physical object from front/top/side perspectives. Determining object-margins in 2D images. Using computer vision techniques to correspond object-margins from different perspectives, and assemble 3D visual representation.

Background:
---
Our idea is coming from a tool called Milkscanner that allows the scanning of objects and creates a displacement map. This displacement map is produced by adding a brunch of cross-sections layer by layer.

Milkscanner is a great idea to produce the displacement map, but it takes a long time to get every cross-sections and some objects are not allowed to put into milk (liquid). Our idea is to generate a displacement map simply with three-views-drawing. We try to use three-views-drawing to determine the position of a point, which is on the surface of our object, in three dimensions space.

![need some pictures](/drawing.png)
*Figure 1. The 3 View and 3D object*

The Challenge:
---
There are following challanges on this project. 
  
    1. The scale of the object on three-views-drawing should be as same as possible to generate an accurate displacement map.
    2. There are no existing research on the same topic that we can refer as a reference, however, some simillar topics exist.
    3. We need to detect the edges as accurate as we can, since the demention factors highly depend on the edges.
    4. We need to learn how to make 3D objects.

Goals and Deliverables:
---
The goals we need to achieve:

    1. Detect the edges of object.
    2. Get rid of unnecessary edges.
    3. Ajusting pictures into same scales.
    4. Mapping demention values togater.
    5. Create 3D objects according to the demention values (x, y, z)
    
The successful project should produce 3D object similar to the original object, except for surfaces that curve into the object. The time should be engough if we start the project early.

*Table 1. Weekly Schedual*
<table style="width:100%">
  <tr>
    <td></td>
    <th>Roy Xu</th>
    <th>Wayne Du</th>
    <th>Jiangyuan Wu</th>
    <th>Zhaohong Wan</th>
  </tr>
  <tr>
    <td>week 1</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>

  </tr>
  <tr>
    <td>week 2</td>
    <td></td>
    <td></td>
    <td></td>
      <td></td>

  </tr>
  <tr>
    <td>week 2</td>
    <td></td>
    <td></td>
    <td></td>
      <td></td>

  </tr>
  <tr>
    <td>week 3</td>
    <td></td>
    <td></td>
    <td></td>
      <td></td>

  </tr>
  <tr>
    <td>week 4</td>
    <td></td>
    <td></td>
    <td></td>
      <td></td>

  </tr>
  <tr>
    <td>week 5</td>
    <td></td>
    <td></td>
    <td></td>
      <td></td>

  </tr>
  <tr>
    <td>week 6</td>
    <td></td>
    <td></td>
    <td></td>
      <td></td>

  </tr>
  <tr>
    <td>week 7</td>
    <td></td>
    <td></td>
    <td></td>
      <td></td>

  </tr>
  <tr>
    <td>week 8</td>
    <td></td>
    <td></td>
    <td></td>
      <td></td>

  </tr>
  <tr>
    <td>week 9</td>
    <td></td>
    <td></td>
    <td></td>
      <td></td>

  </tr>
</table>
