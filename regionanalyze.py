import numpy as np
import mca
import json

class RegionAnalyzer:
    def __init__(self, file, world):
        self.region = mca.Region.from_file(file)
        with open("./encoding.json") as f:
            self.encode_sequence = json.load(f)
            self.name_sequence = self.encode_sequence['name']
            self.empty = np.zeros((1,12),dtype=np.float32)
        self.world = world
        self.file = file
    
    def encode_block(self, block:mca.Block,coords:tuple[3]):
        '''
        encode_block Docstring
        x, y, z, are ralative to the start of section
        transform the block into a 12 dim list(not include coords)
        transform the coords into 6 dim list and join them
        return: np.array(1,12)
        '''
        name = block.id
        if name == 'air':
            return np.zeros((1,12),np.float32)
        properties = block.properties
        encode = []
        #process the key related to name
        #model
        for j in self.encode_sequence["model"]:
            if j in name:
                encode.append(self.encode_sequence["model"][j])
                break
        #texture
        for j in self.name_sequence['natrue']:
            if j in name:
                encode.append(self.name_sequence['natrue'][j])
        for j in self.name_sequence['metal']:
            if j in name:
                encode.append(self.name_sequence['metal'][j])
                break
        def encoder(s):
            for j in self.name_sequence[s][0]:
                if j in name:
                    for k in self.name_sequence[s][1]:
                        if k in name:
                            return self.name_sequence[s][0][j].append(
                                   self.name_sequence[s][1][k])
        _ = encoder("coloured_block")
        if _ != None:
            encode.append(_)
        _ = encoder("wood")
        if _ != None:
            encode.append(_)
        _ = encoder("stone")
        if _ != None:
            encode.append(_)
        if encode == None:
            return[0 for i in range(6)]#fill 'air' instead
        #states
        for j in self.encode_sequence['states']:
            for k in j:
                if k in properties:
                    encode.append(self.encode_sequence['states'][j][k])
                    break
                encode.append(0)
        #coords
        if len(coords) != 3:
            raise ValueError("Invalid coords")
        encode.append([divmod(i,4) for i in coords])
        return np.array(encode,dtype=np.float32)

    def encode_chunk(self, x, z):
        chunk = self.region.get_chunk(x, z)
        data = []
        gen  = chunk.stream_chunk()
        i = 0
        while True:
            Y = i % 4096
            y,z = divmod(Y,256)
            z = z // 16
            x = Y % 16
            try:
                data.append(self.encode_block(next(gen),(x,y,z)))
            except StopIteration:
                break
            i += 1
        return np.array(data)
        
    def encode_region(self):
        region = []
        for i in range(32):
            for j in range(32):
                region.append(self.encode_chunk(i,j))
        np.savez(f"./data/{self.world}/{self.file[:-3]}npz")
