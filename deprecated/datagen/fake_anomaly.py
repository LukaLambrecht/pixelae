import numpy as np
rng = np.random.default_rng()


def rectangle_mask(shape,
    rectangle_anchor='random',
    rectangle_shape='random',
    rectangle_min_pixels=None,
    shape_mask=None):
    # mask a rectangle with specified location and size
    mask = np.zeros(shape).astype(bool)
    if rectangle_anchor=='random':
        x = rng.integers(low=0, high=shape[0]) # random position
        y = rng.integers(low=0, high=shape[1]) # random position
    else: x,y = rectangle_anchor
    if rectangle_shape=='random':
        dx = rng.integers(low=1, high=int(shape[0]/5.)) # random between 1 and a fraction of the width
        dy = rng.integers(low=1, high=int(shape[1]/5.)) # random between 1 and a fraction of the height
    else: dx,dy = rectangle_shape
    if(rectangle_min_pixels is not None and dx*dy < rectangle_min_pixels):
        return rectangle_mask(shape, rectangle_anchor=rectangle_anchor,
                              rectangle_shape=rectangle_shape, 
                              rectangle_min_pixels=rectangle_min_pixels,
                              shape_mask=shape_mask)
    if(x+dx >= shape[0] or y+dy >= shape[1]):
        return rectangle_mask(shape, rectangle_anchor=rectangle_anchor,
                              rectangle_shape=rectangle_shape, 
                              rectangle_min_pixels=rectangle_min_pixels,
                              shape_mask=shape_mask)
    mask[x:x+dx, y:y+dy] = True
    paramdict = {'rectangle_anchor': (x,y), 'rectangle_shape': (dx,dy)}
    if shape_mask is not None:
        if np.any((mask & ~shape_mask)):
            return rectangle_mask(shape, rectangle_anchor=rectangle_anchor,
                                  rectangle_shape=rectangle_shape,
                                  rectangle_min_pixels=rectangle_min_pixels,
                                  shape_mask=shape_mask)
    return (mask, paramdict)

def sector_mask(shape,
    central_angle='random',
    opening_angle='random',
    inner_radius='random',
    shape_mask=None):
    # mask a circle sector with specified angle, opening angle and radius
    # reference: https://www.appsloveworld.com/numpy/100/7/mask-a-circular-sector-in-a-numpy-array?expand_article=1
    
    # make central angle and opening angle
    if central_angle=='random': central_angle = rng.random()*2*np.pi # random between 0 and 360 degrees
    if opening_angle=='random': opening_angle = rng.random()*np.pi/2. # random between 0.1 and 90 degrees
    min_angle = central_angle - opening_angle/2.
    max_angle = central_angle + opening_angle/2.
    min_angle %= (2*np.pi)
    max_angle %= (2*np.pi)
    
    # make center coordinates
    if inner_radius=='random': inner_radius = rng.random()*shape[0]/3. # random between 0 and 2/3 of radius
    cx = shape[0]/2. + inner_radius*np.cos(central_angle)
    cy = shape[1]/2. + inner_radius*np.sin(central_angle)

    # convert cartesian --> polar coordinates
    x,y = np.ogrid[:shape[0],:shape[1]]
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(y-cy, x-cx)
    theta %= (2*np.pi)

    # angular mask
    mask = ((theta<max_angle) & (theta>min_angle))
    if max_angle<min_angle: mask = ~((theta>max_angle) & (theta<min_angle))
    paramdict = {'central_angle': central_angle, 'opening_angle': opening_angle, 'inner_radius': inner_radius}

    if shape_mask is not None:
        if np.all(((mask) & (~shape_mask))):
            # todo: the recursion below will never converge,
            # as all parameters are fixed at this stage
            return sector_mask(shape, central_angle=central_angle,
                               opening_angle=opening_angle,
                               inner_radius=inner_radius,
                               shape_mask=shape_mask)
    return (mask, paramdict)

def dead_pixel():
    pass

def dead_rectangle(hists, **kwargs):
    (mask, paramdict) = rectangle_mask((hists.shape[1],hists.shape[2]), **kwargs)
    anomalous_hists = np.copy(hists)
    anomalous_hists[:,mask] = 0
    return (anomalous_hists, paramdict)

def dead_sector(hists, **kwargs):
    (mask, paramdict) = sector_mask((hists.shape[1],hists.shape[2]), **kwargs)
    anomalous_hists = np.copy(hists)
    anomalous_hists[:,mask] = 0
    return (anomalous_hists, paramdict)

def hot_pixel():
    pass

def hot_rectangle(hists, hotfactor=5, **kwargs):
    (mask, paramdict) = rectangle_mask((hists.shape[1],hists.shape[2]), **kwargs)
    anomalous_hists = np.copy(hists)
    anomalous_hists[:,mask] = hotfactor*anomalous_hists[:,mask]
    return (anomalous_hists, paramdict)

def hot_sector(hists, hotfactor=5, **kwargs):
    (mask, paramdict) = sector_mask((hists.shape[1],hists.shape[2]), **kwargs)
    anomalous_hists = np.copy(hists)
    anomalous_hists[:,mask] = hotfactor*anomalous_hists[:,mask]
    return (anomalous_hists, paramdict)