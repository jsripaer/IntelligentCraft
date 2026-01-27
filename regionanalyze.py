import numpy as np
import mca#This library is officially called mcapy.
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
            return np.zeros((1,18),np.float32)
        properties = block.properties
        encode = []
        if name == 'gravel':
            pass
        #process the key related to name
        #model
        for j in self.encode_sequence["model"]:
            if j in name:
                encode.extend(self.encode_sequence["model"][j])
                break
        if encode == []:
            encode = [0,0,0]
        #texture
        for j in self.name_sequence['natrue']:
            if j in name:
                encode.extend(self.name_sequence['natrue'][j])
                break
            else:
                for j in self.name_sequence['metal']:
                    if j in name:
                        encode.extend(self.name_sequence['metal'][j])
                        break
        def encoder(s):
            for j in self.name_sequence[s][0]:
                if j in name:
                    for k in self.name_sequence[s][1]:
                        if k in name:
                            return self.name_sequence[s][0][j].extend(
                                   self.name_sequence[s][1][k])
        _ = encoder("coloured_block")
        if _ != None:
            encode.extend(_)
        else:
            _ = encoder("wood")
            if _ != None:
                encode.extend(_)
            else:
                _ = encoder("stone")
                if _ != None:
                    encode.extend(_)
        if encode == [0,0,0]:
            encode.extend([0,0,0])#fill 'air' instead
        #states
        for j in self.encode_sequence['state']:
            l = len(encode)
            for k in self.encode_sequence['state'][j]:
                if k in properties:
                    encode.append(self.encode_sequence['state'][j][k])
                    break
            if len(encode) == l:
                encode.append(0)
        #coords
        for i in range(3):
            encode.extend(divmod(coords[i],4))
        return np.array(encode,dtype=np.float32).reshape((1,18))
    
    def encode_section(self, chunk, Y):

        sec = chunk.stream_blocks(Y)
        data0 = self.encode_block(next(sec),coords=(0,0,0))
        i = 1
        while i < 4096:
            y,z = divmod(i,256)
            z = z // 16
            x = i % 16
            try:
                data0 = np.vstack((data0,self.encode_block(next(sec),(x,y,z))))
            except StopIteration:
                break
            except IndexError:
                data0 = np.vstack((data0,np.zeros((1,18))))
            i += 1
        m = 4096 - data0.shape[0]
        if m > 0:
            data0 = np.vstack((data0,np.zeros((m,18))))
        return data0.reshape((1,4096,18))

    def encode_chunk(self, x, z):
        i = 0
        chunk = self.region.get_chunk(x,z)
        data = self.encode_section(chunk,-4)
        for Y in range(-3,19):
            data = np.vstack((data,self.encode_section(chunk , Y)))
        
    def encode_region(self):
        region = []
        for i in range(32):
            for j in range(32):
                region.append(self.encode_chunk(i,j))
        np.savez(f"./data/{self.world}/{self.file[:-3]}npz")

