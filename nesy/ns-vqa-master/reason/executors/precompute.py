import pickle

properties = {
  "shape": {
    "cube": "SmoothCube_v2",
    "sphere": "Sphere",
    "cylinder": "SmoothCylinder"
  },
  "color": {
    "gray": [87, 87, 87],
    "red": [173, 35, 35],
    "blue": [42, 75, 215],
    "green": [29, 105, 20],
    "brown": [129, 74, 25],
    "purple": [129, 38, 192],
    "cyan": [41, 208, 208],
    "yellow": [255, 238, 51]
  },
  "material": {
    "rubber": "Rubber",
    "metal": "MyMetal"
  },
  "size": {
    "large": 0.6,
    "small": 0.3
  }
}


region_constraint = pickle.load(open( "/content/drive/MyDrive/ColabNotebooks/clevr-abductive/nesy/ns-vqa-master/reason/executors/consistent_combination.p", "rb" ))
a = ["color", "material", "shape", "size"]
#key = (cid, feature, val) --> list of regions where feature = val


dict_cfv = {} 
for (cid, rid) in region_constraint:
  consis = region_constraint[(cid, rid)]
  for f in properties:
    ind = a.index(f)
    values = properties[f]
    for v in values:
      for c in consis:
        if c[ind] == v:
          if (cid, f, v) not in dict_cfv:
            dict_cfv[(cid, f, v)] = [rid]
          else:
            if rid not in dict_cfv[(cid, f, v)]:
              dict_cfv[(cid, f, v)].append(rid)
          break    

print("Dict_cfv::")
print(dict_cfv)
pickle.dump(dict_cfv, open("/content/drive/MyDrive/ColabNotebooks/clevr-abductive/nesy/ns-vqa-master/reason/executors/dict_cfv.p", "wb" ) )

dict_crf = {}
for (cid, rid) in region_constraint:
  consis = region_constraint[(cid, rid)]
  for f in properties:
    ind = a.index(f)
    for c in consis:
      if (cid, rid, f) not in dict_crf:
        dict_crf[(cid, rid, f)] = [c[ind]]
      else:
        if c[ind] not in dict_crf[(cid, rid, f)]:
              dict_crf[(cid, rid, f)].append(c[ind]) 
print("Dict_crf::")
print(dict_crf)

pickle.dump(dict_crf, open("/content/drive/MyDrive/ColabNotebooks/clevr-abductive/nesy/ns-vqa-master/reason/executors/dict_crf.p", "wb" ) )
