import json
import os
from abc import ABC, abstractmethod
import logging
import torch
from rdkit import Chem
from reinvent_scoring.scoring.enums.scoring_function_component_enum import ScoringFunctionComponentNameEnum
from reinvent_scoring.scoring.score_summary import FinalSummary

from running_modes.configurations import GeneralConfigurationEnvelope, ReinforcementLoggerConfiguration


class BaseReinforcementLogger(ABC):
    def __init__(self, configuration: GeneralConfigurationEnvelope, log_config: ReinforcementLoggerConfiguration):
        self._log_config = log_config
        self._configuration = configuration
        self._setup_workfolder()
        self._logger = self._setup_logger()

    @abstractmethod
    def log_message(self, message: str):
        raise NotImplementedError("log_message method is not implemented")

    @abstractmethod
    def timestep_report(self, start_time, n_steps, step, score_summary: FinalSummary,
                        agent_likelihood: torch.tensor, prior_likelihood: torch.tensor,
                        augmented_likelihood: torch.tensor, diversity_filter, agent):
        raise NotImplementedError("timestep_report method is not implemented")

    def log_out_input_configuration(self):
        file = os.path.join(self._log_config.result_folder, "input.json")
        jsonstr = json.dumps(self._configuration, default=lambda x: x.__dict__, sort_keys=True, indent=4,
                             separators=(',', ': '))
        with open(file, 'w') as f:
            f.write(jsonstr)

    def save_checkpoint(self, step: int, diversity_filter, agent):
        actual_step = step + 1
        if self._log_config.logging_frequency > 0 and actual_step % self._log_config.logging_frequency == 0:
            self.save_filter_memory(diversity_filter)
            agent.save_to_file(os.path.join(self._log_config.result_folder, f'Agent.{actual_step}.ckpt'))

    def save_filter_memory(self, diversity_filter):
        diversity_memory = diversity_filter.get_memory_as_dataframe()
        # TODO: Pass a job_name parameter from the config
        self.save_to_csv(diversity_memory, self._log_config.result_folder)

    def save_to_csv(self, scaffold_memory, path, job_name="default_job"):
        sf_enum = ScoringFunctionComponentNameEnum()

        if not os.path.isdir(path):
            os.makedirs(path)
        file_name = os.path.join(path, "scaffold_memory.csv")
        if len(scaffold_memory) > 0:
            sorted_df = scaffold_memory.sort_values(sf_enum.TOTAL_SCORE, ascending=False)
            smiles = sorted_df['SMILES'].to_list()
            #print(smiles)
            smiles_macro=[]
            dele_idxs = []
            for idx, smi in enumerate(smiles):
                mol = Chem.MolFromSmiles(smi)
                atoms = mol.GetAtoms()
                indexs_F=[]
                indexs_Cl=[]
                for index,atom in enumerate(atoms):
                    if atom.GetAtomicNum() == 9:
                        indexs_F.append(index)
                    if atom.GetAtomicNum() == 17:
                        indexs_Cl.append(index)
                if len(indexs_Cl) == 2 and len(indexs_F) == 2:
                    distance_F = (indexs_F[1] - indexs_F[0])**2
                    distance_Cl = (indexs_Cl[1] - indexs_Cl[0])**2
                    if distance_F < distance_Cl:
                        indexs_F = []
                    else:
                        indexs_Cl = []
                if len(indexs_Cl) == 2 and len(indexs_F) < 2:
                    neighbor1 = atoms[indexs_Cl[0]].GetNeighbors()
                    neighbor2 = atoms[indexs_Cl[1]].GetNeighbors()
                    neighbors = neighbor1 + neighbor2
                    if neighbors[0].GetSymbol() == 'C' and not neighbors[0].IsInRing():
                        neighbor_sub = neighbors[0].GetNeighbors()
                        for sub in neighbor_sub:
                            if sub.GetSymbol() != 'Cl':
                                indexs_Cl.append(sub.GetIdx())
                        indexs_Cl.append(neighbors[1].GetIdx())
                    else:
                        indexs_Cl.append(neighbors[1].GetIdx() - 1)
                        indexs_Cl.append(neighbors[0].GetIdx())
                    mw = Chem.RWMol(mol)
                    mw.AddBond(indexs_Cl[2], indexs_Cl[3], Chem.BondType.SINGLE)
                    m_edit = mw.GetMol()
                    patt1 = Chem.MolFromSmarts('[CD2](-[#17])')
                    patt2 = Chem.MolFromSmarts('Cl')
                    m = Chem.DeleteSubstructs(m_edit, patt1)
                    macro = Chem.DeleteSubstructs(m, patt2)
                    smile = Chem.MolToSmiles(macro)
                    smiles_macro.append(smile)



                elif len(indexs_F) == 2:
                    neighbor1 = atoms[indexs_F[0]].GetNeighbors()
                    neighbor2 = atoms[indexs_F[1]].GetNeighbors()
                    neighbors = neighbor1 + neighbor2
                    if neighbors[0].GetSymbol() == 'C' and not neighbors[0].IsInRing():
                        neighbor_sub = neighbors[0].GetNeighbors()
                        for sub in neighbor_sub:
                            if sub.GetSymbol() != 'F':
                                indexs_F.append(sub.GetIdx())
                        indexs_F.append(neighbors[1].GetIdx())
                    else:
                        indexs_F.append(neighbors[1].GetIdx() - 1)
                        indexs_F.append(neighbors[0].GetIdx())
                    mw = Chem.RWMol(mol)
                    mw.AddBond(indexs_F[2], indexs_F[3], Chem.BondType.SINGLE)
                    m_edit = mw.GetMol()
                    patt1 = Chem.MolFromSmarts('[CD2](-[#9])')
                    patt2 = Chem.MolFromSmarts('F')
                    m = Chem.DeleteSubstructs(m_edit, patt1)
                    macro = Chem.DeleteSubstructs(m, patt2)
                    smile = Chem.MolToSmiles(macro)
                    smiles_macro.append(smile)
                else:
                    dele_idxs.append(idx)
            sorted_df = sorted_df.drop(dele_idxs)
            sorted_df["SMILES"] = smiles_macro
            sorted_df["ID"] = [f"{job_name}_{e}" for e, _ in enumerate(sorted_df.index.array)]
            sorted_df.to_csv(file_name, index=False)

    def _setup_workfolder(self):
        if not os.path.isdir(self._log_config.logging_path):
            os.makedirs(self._log_config.logging_path)
        if not os.path.isdir(self._log_config.result_folder):
            os.makedirs(self._log_config.result_folder)

    def _setup_logger(self):
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s: %(module)s.%(funcName)s +%(lineno)s: %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger("reinforcement_logger")
        if not logger.handlers:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger
