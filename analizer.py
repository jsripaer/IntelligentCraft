import zlib
import json
import numpy as np

#This is a light weight NBT parser.
#It just extracts the data from the NBT file, then returns the Numpy array for training.

f = open("D:\\IntelligentCraft\\IntelligentCraft\\encoding.json")
encode_sequence = json.load(f)
name_sequence = encode_sequence["name"]
f.close()
class analizer:
    def __init__(self, file="./test.mca", data: bytes=None):
        if data is None:
            with open(file, 'rb') as f:
                data = f.read()
                self.data = data
        else:
            self.data = data

    def chunkData(self,x,z)->bytes:
        off = 4 * (x % 32 + z % 32 * 32)
        off = int.from_bytes(self.data[off : off + 3],byteorder='big')
        if off == 0 and self.data[off + 3] == 0:
            return None#this means the chunk at given coordinate has'n been generated yet
        off *= 4096
        length = int.from_bytes(self.data[off : off + 4],byteorder='big')
        if self.data[off + 4] == 1:
            raise BufferError("Gzip is not supported")
        return zlib.decompress(self.data[off + 5 : off + 5 + length - 1])

    @staticmethod
    def findAll(string, data:bytes=None)->list:
        '''
        search the data with given string
        desinged to avoid analize all the tags
        return : list of indexs 
        '''
        if data is None:
            raise ValueError("data can't be None")
        _range = len(data) - len(string)
        return [i for i in range(_range) if string.encode() == data[i:i+len(string)]]
    
    def paletteMap(self,data:bytes)->np.ndarray:
        '''
        this method will return the compund list of encoded properties 
        data: the palette of one setion, starts from block_states to biome, nbt tag or json bytes?
        return: an encoded numpy matrix in the sequence of palette
        '''
        #this file should be rewrite in C or Cython to avoid serious performance difficulty 
        if data is None:
            return [0 for i in range(10)]#air, the section is empty
        
        indexs = self.findAll('Name')
        code_list = [[] for i in range(len(indexs))]
        #encode logic has to be written by hand, which is really a frustrating thing.
        for i in range(len(indexs)): 
            index = indexs[i]
            name_length = int.from_bytes(data[index+4 : index+6])
            name = data[index+16 : index+6+name_length].decode()
            #process the key related to name
            #model
            for j in encode_sequence["model"]:
                if j in name:
                    code_list[i].append(encode_sequence["model"][j])
            #texture
            if name in name_sequence["nature"]:
                code_list[i].append(name_sequence["nature"][name])
                continue
            if name in name_sequence["metal"]:
                code_list[i].append(name_sequence["metal"])
                continue
            def encode(s):
                for j in name_sequence[s][0]:
                    if j in name:
                        for k in name_sequence[s][1]:
                            if k in name:
                                return name_sequence[s][0][j].append(
                                       name_sequence[s][1][k])
            _ = encode("coloured_block")
            if _ != None:
                code_list[i] = _
                continue
            _ = encode("wood")
            if _ != None:
                code_list[i] = _
                continue
            _ = encode("stone")
            if _ != None:
                code_list[i] = _
                continue
            code_list[i] == []#the block isn't in the encoding_sequence
            code_list[i] = [0 for i in range(10)]#fill 'air' instead
            
            #state
            properties = data[indexs[i]:indexs[i+1]]
            #to make sure states won't affect each other
            #in the first loop, find the indexs of all the states
            #as the aobve code has excluded many illegal states, there won't be error
            state_indexs = []
            for j in encode_sequence["state"]:
                state_indexs.append(self.findall(j,properties)[0])
            #then, encode the states 
            for j in state_indexs:
                for k in encode_sequence["state"][j]:
                    if k in properties[j:state_indexs[j+1]]:
                        code_list[i].extend(encode_sequence["state"][j][k])
        code_list = np.array(code_list)
        return code_list
    
    def chunkMap(self,x,z)->np.ndarray:
        data = self.chunkData(x,z)
        if data is None:
            return None
        chunk = np.zeros((32,4096,12),dtype=np.float32)
        section_indexs = self.findAll('BlockStates',data)
        for i in range(len(section_indexs)):
            section = data[section_indexs[i]:section_indexs[i+1]]
            chunk[i] = self.paletteMap(section)
        return chunk
    
    def regionMap(self,region_x,region_z):
        '''
        return the whole region data as a numpy array
        shape: (1024,4096,12)
        '''
        region = np.zeros((1024,4096,12),dtype=np.float32)
        for i in range(32):
            for j in range(32):
                chunk = self.chunkMap(region_x*32 + i, region_z*32 + j)
                if chunk is not None:
                    yield region
    
    def saveRegion(self, region_x, region_z, path):
        '''
        give the coordinate or region file path to save the region data as a numpy file
        1024 chunks in a file, saved with np.savez so that it can be loaded easily
        '''
        region_data = self.regionMap(region_x, region_z)
        np.savez(path, region_data)
    
    @staticmethod
    def blockMapfromJson(self,data:bytes):
        '''
        json file should be a sequence of blocks(4096)
        each block should be a dict including 'name' and 'states'
        '''
        json_data = json.loads(data.decode())
        code_list = []
        for block in json_data:
            name = block['name']
            code_block = []
            #model
            for j in encode_sequence["model"]:
                if j in name:
                    code_block.append(encode_sequence["model"][j])
            #texture
            if name in name_sequence["nature"]:
                code_block.append(name_sequence["nature"][name])
            elif name in name_sequence["metal"]:
                code_block.append(name_sequence["metal"])
            else:
                def encode(s):
                    for j in name_sequence[s][0]:
                        if j in name:
                            for k in name_sequence[s][1]:
                                if k in name:
                                    return name_sequence[s][0][j].append(
                                           name_sequence[s][1][k])
                _ = encode("coloured_block")
                if _ != None:
                    code_block = _
                else:
                    _ = encode("wood")
                    if _ != None:
                        code_block = _
                    else:
                        _ = encode("stone")
                        if _ != None:
                            code_block = _
                        else:
                            code_block == []#the block isn't in the encoding_sequence
                            code_block = [0 for i in range(10)]#fill 'air' instead
            #state
            properties = block['states']
            for j in properties:
                if j in encode_sequence["state"]:
                    if properties[j] in encode_sequence["state"][j]:
                        code_block.extend(encode_sequence["state"][j][properties[j]])
            code_list.append(code_block)
        code_list = np.array(code_list)
        return code_list