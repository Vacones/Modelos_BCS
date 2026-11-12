"""Asociación única de vaca y trasero a partir de cajas predichas."""


def pair_boxes(boxes, coverage=.8):
    """Solo un par único: sin usar las cajas reales ni el BCS para escoger."""
    cows=[b for b in boxes if b['class']==0];rumps=[b for b in boxes if b['class']==1];pairs=[]
    for cow in cows:
        a,b,c,d=cow['xyxy']
        for rump in rumps:
            x,y,z,w=rump['xyxy']; area=(z-x)*(w-y)
            if area<=0:continue
            intersect=max(0,min(c,z)-max(a,x))*max(0,min(d,w)-max(b,y))
            if a<=(x+z)/2<=c and b<=(y+w)/2<=d and intersect/area>=coverage:
                pairs.append((cow,rump))
    if not pairs:return 'sin_par_valido',None
    if len(pairs)>1:return 'asociacion_ambigua',None
    return 'ok',pairs[0]
