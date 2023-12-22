"""
@author: BrozosCh
"""
import os
import os.path as osp
import time
import torch
import torch.nn.functional as F
from torch_sparse import coalesce
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)

try:
    import rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import rdBase
    from rdkit.Chem.rdchem import HybridizationType
    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures
    from rdkit.Chem.rdchem import BondType as BT
    from rdkit.Chem import Draw
    rdBase.DisableLog('rdApp.error')
except ImportError:
    rdkit = None


class OwnDataset(InMemoryDataset):


    raw_url = ''
    processed_url = ''


    if rdkit is not None:
        types = {'C' :0 , 'N' : 1, 'O' : 2, 'S' : 3, 'F' :4, 'Cl' : 5, 'Br' :6, 'Na' : 7, 'I': 8, 'B' :9, 'K' :10, 'H' :11, 'Li' :12} # atom types
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3} #bond types

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        super(OwnDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'raw.pt' if rdkit is None else 'raw.csv'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

        #url = self.processed_url if rdkit is None else self.raw_url
        #file_path = download_url(url, self.raw_dir)
        #extract_zip(file_path, self.raw_dir)
        #os.unlink(file_path)

    def process(self):
        if rdkit is None:
            print('Using a pre-processed version of the dataset. Please '
                  'install `rdkit` to alternatively process the raw data.')

            self.data, self.slices = torch.load(self.raw_paths[0])
            data_list = [data for data in self]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            return

        molecules = []
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1] #skip first line
            molecules = [[str(x) for x in line.split(";")[0:1]] for line in data]    #('<separator of letter>') in split, whitespace corresponds to blank parentheses 
            writer = Chem.SDWriter(str(self.root) + '/raw/raw.sdf')
            for k,m in enumerate(molecules):
                mol = Chem.rdmolfiles.MolFromSmiles(m[0])
                print(k)
                #mol = Chem.rdmolops.AddHs(mol)   # explicit trivial Hs (excluded)
                writer.write(mol)
            del writer


            target = [[float(x) for x in line.split(";")[1:2]]    #('<separator of letter>') in split, whitespace corresponds to blank parentheses, [<number of targets>]
                      for line in data]
            target = torch.tensor(target, dtype=torch.float)

        # delay for proper saving of sdf file
        time.sleep(10)
            
        dataset = str(self.root) + '/raw/raw.sdf'
        suppl = Chem.SDMolSupplier(dataset, removeHs=False)
        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

        data_list = []
        max_len = 0
        max_name = ''
        if len(target) == len(suppl):
            print('True')
        else:
            print('Fail: number of target data points does not match number of molecules')

        for i, mol in enumerate(suppl):
            if mol is None:
                print('Invalid molecule (None)')
                continue

            text = suppl.GetItemText(i)
            N = mol.GetNumAtoms()

            # atom features
            type_idx = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            sp3d = []
            sp3d2 = []
            num_hs = []
            num_neighbors = []
            for atom in mol.GetAtoms():
                type_idx.append(self.types[atom.GetSymbol()])
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
                sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)
                sp3d2.append(1 if hybridization == HybridizationType.SP3D2 else 0)              
                num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))
                num_neighbors.append(len(atom.GetNeighbors()))                

            x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(self.types))
            x2 = torch.tensor([aromatic, sp, sp2, sp3, sp3d, sp3d2], dtype=torch.float).t().contiguous()
            x3 = F.one_hot(torch.tensor(num_neighbors), num_classes=6)
            x4 = F.one_hot(torch.tensor(num_hs), num_classes=5)
            x = torch.cat([x1.to(torch.float), x2, x3.to(torch.float),x4.to(torch.float)], dim=-1)

           
            row, col, bond_idx, conj, ring, stereo = [], [], [], [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                bond_idx += 2 * [self.bonds[bond.GetBondType()]]
                conj.append(bond.GetIsConjugated())
                conj.append(bond.GetIsConjugated())
                ring.append(bond.IsInRing())
                ring.append(bond.IsInRing())
                stereo.append(bond.GetStereo())
                stereo.append(bond.GetStereo())

            edge_index = torch.tensor([row, col], dtype=torch.long)
            e1 = F.one_hot(torch.tensor(bond_idx),num_classes=len(self.bonds)).to(torch.float)
            e2 = torch.tensor([conj, ring], dtype=torch.float).t().contiguous()
            e3 = F.one_hot(torch.tensor(stereo),num_classes=6).to(torch.float)
            edge_attr = torch.cat([e1, e2, e3], dim=-1)
            edge_index, edge_attr = coalesce(edge_index, edge_attr, N, N)

            # target data
            mol_id = i+1
            y = target[i].unsqueeze(0)

            # transform SMILES into ascii data type and store it in a name torch tensor
            name = str(Chem.MolToSmiles(mol))
            ascii_name = []
            for c in name:
                ascii_name.append(int(ord(c)))

            if len(ascii_name) > max_len:
                max_len = len(ascii_name)
                max_name = name

            ## if fails, increase range
            for i in range(len(ascii_name), 300):
                ascii_name.append(0)

            ascii_name = torch.tensor([ascii_name], dtype=torch.float).contiguous()

            # print current molecule with target data
            print(str(name) + ': ' + str(y.item()))

            # save data
            data = Data(x=x, edge_attr = edge_attr, edge_index=edge_index,  y=y, mol_id=ascii_name)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

