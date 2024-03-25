import time
from rdkit import Chem
import numpy as np

from reinvent_scoring.scoring.score_summary import FinalSummary
from reinvent_chemistry.logging import fraction_valid_smiles


class ConsoleMessage:

    def create(self, start_time, n_steps, step, score_summary: FinalSummary,
               agent_likelihood, prior_likelihood, augmented_likelihood):
        mean_score = np.mean(score_summary.total_score)
        time_message = self._time_progress(start_time, n_steps, step, score_summary.scored_smiles, mean_score)
        score_message = self._score_profile(score_summary.scored_smiles, agent_likelihood, prior_likelihood,
                                            augmented_likelihood, score_summary.total_score)
        score_breakdown = self._score_summary_breakdown(score_summary)
        message = time_message + score_message + score_breakdown
        return message

    def _time_progress(self, start_time, n_steps, step, smiles, mean_score):
        time_elapsed = int(time.time() - start_time)
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        valid_fraction = fraction_valid_smiles(smiles)
        message = (f"\n Step {step}   Fraction valid SMILES: {valid_fraction:4.1f}   Score: {mean_score:.4f}   "
                   f"Sample size: {len(smiles)}   "
                   f"Time elapsed: {time_elapsed}   "
                   f"Time left: {time_left:.1f}\n")
        return message

    def _score_profile(self, smiles, agent_likelihood, prior_likelihood, augmented_likelihood, score):
        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()
        smiles_macro=[]
        for smi in smiles:
            #print(smi)
            mol = Chem.MolFromSmiles(smi)
            if mol:
                atoms = mol.GetAtoms()
                indexs_F=[]
                indexs_Cl=[]
            else:
                smiles_macro.append(smi)
                continue
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
                smile = Chem.MolToSmiles(mw)
                #print(smil)
                try:
                    mw.AddBond(indexs_Cl[2], indexs_Cl[3], Chem.BondType.SINGLE)
                    m_edit = mw.GetMol()

                    patt1 = Chem.MolFromSmarts('[CD2](-[#17])')
                    patt2 = Chem.MolFromSmarts('Cl')
                    m = Chem.DeleteSubstructs(m_edit, patt1)
                    macro = Chem.DeleteSubstructs(m, patt2)
                    smile = Chem.MolToSmiles(macro)
                except:
                    pass    
                #print(smile)
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
                smil = Chem.MolToSmiles(m_edit)
                #print(smil)
                patt1 = Chem.MolFromSmarts('[CD2](-[#9])')
                patt2 = Chem.MolFromSmarts('F')
                m = Chem.DeleteSubstructs(m_edit, patt1)
                macro = Chem.DeleteSubstructs(m, patt2)
                smile = Chem.MolToSmiles(macro)
                smiles_macro.append(smile)
            else:
                smiles_macro.append(smi)
        message = "     ".join(["  Agent", "Prior", "Target", "Score"] + ["SMILES\n"])
        for i in range(min(10, len(smiles_macro))):
            message += f'{agent_likelihood[i]:6.2f}    {prior_likelihood[i]:6.2f}    ' \
                       f'{augmented_likelihood[i]:6.2f}    {score[i]:6.2f} '
            message += f"     {smiles_macro[i]}\n"
        return message

    def _score_summary_breakdown(self, score_summary: FinalSummary):
        message = "   ".join([c.name for c in score_summary.profile])
        message += "\n"
        for i in range(min(10, len(score_summary.scored_smiles))):
            for summary in score_summary.profile:
                message += f"{summary.score[i]}   "
            message += "\n"
        return message
