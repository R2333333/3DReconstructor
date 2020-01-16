# 3DReconstructor

## Construction of 3D Visual Representations From 2D Images


Authors:
---
Roy Xu, Jiangyuan Wu, Wayne Du


Description:
---
With regular cameras, capturing 2D images of a physical object from front/top/side perspectives. Determining object-margins in 2D images. Using computer vision techniques to martch object-margins from different perspectives, and assemble 3D visual representation.

Background:
---
Our idea is coming from a tool called Milkscanner that allows the scanning of objects and creates a displacement map. This displacement map is produced by adding a brunch of cross-sections layer by layer.

Milkscanner is a great idea to produce the displacement map, but it takes a long time to get every cross-sections and some objects are not allowed to put into milk (liquid). Our idea is to generate a displacement map simply with three-views-drawing. We try to use three-views-drawing to determine the position of a point, which is on the surface of our object, in three dimensions space.

![need some pictures](/drawing.png)

The Challenge:
---
There are following challanges on this project. 
  
    1. The scale of the object on three-views-drawing should be as same as possible to generate an accurate displacement map.
    2. There are no existing research on the same topic that we can refer as a reference, however, some simillar topics exist.
    3. We need to detect the edges as accurate as we can, since the demention factors highly depend on the edges.
    3. We need to learn how to make 3D objects.
    

