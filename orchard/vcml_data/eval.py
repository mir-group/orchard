import os
import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

def check_physi_or_chemi(sys):
    # distinguish between chemisorption and physiorption in ADS41
    if sys in ['C6H6+Ag111->C6H6@Ag111',
               'C6H6+Au111->C6H6@Au111',
               'C6H6+Cu111->C6H6@Cu111',
               'C6H6+Pt111->C6H6@Pt111',
               'C2H6+Pt111->C2H6@Pt111',
               'C3H8+Pt111->C3H8@Pt111',
               'C4H10+Pt111->C4H10@Pt111',
               'CH3I+Pt111->CH3I@Pt111',
               'CH4+Pt111->CH4@Pt111',
               'C6H10+Pt111->C6H10@Pt111',
               'H2O+O@Pt111->H2O-OD@Pt111',
               'H2O+Pt111->H2O@Pt111',
               'CH3OH+Pt111->CH3OH@Pt111',
               'C10H8+Pt111->C10H8@Pt111',
               'NH3+Cu100->NH3@Cu100']:
        physi = True
    else:
        physi = False
    return physi


def rename_dbh24(sys):
    dbh24_names = {'r1forward':r'H$^{\bullet}$ + N$_{2}$O $\rightarrow$ OH$^{\bullet}$ + N$_{2}$',
                   'r1reverse':r'H$^{\bullet}$ + N$_{2}$O $\leftarrow$ OH$^{\bullet}$ + N$_{2}$',
                   'r2forward':r'H$^{\bullet}$ + ClH $\rightarrow$ H$^{\bullet}$ + HCl',
                   'r2reverse':r'H$^{\bullet}$ + ClH $\leftarrow$ H$^{\bullet}$ + HCl',
                   'r3forward':r'CH$_{3}^{\bullet}$ + FCl $\rightarrow$ CH$_{3}$F + Cl$^{\bullet}$',
                   'r3reverse':r'CH$_{3}^{\bullet}$ + FCl $\leftarrow$ CH$_{3}$F + Cl$^{\bullet}$',
                   'r4forward':r'Cl$^{-}$ + CH$_{3}$Cl $\rightarrow$ CH$_{3}$Cl + Cl$^{-}$',
                   'r4reverse':r'Cl$^{-}$ + CH$_{3}$Cl $\leftarrow$ CH$_{3}$Cl + Cl$^{-}$',
                   'r5forward':r'CH$_{3}$Cl + F$^{-}$ $\rightarrow$ CH$_{3}$OH + F$^{-}$',
                   'r5reverse':r'CH$_{3}$Cl + F$^{-}$ $\leftarrow$ CH$_{3}$OH + F$^{-}$',
                   'r6forward':r'OH$^{-}$ + CH$_{3}$F $\rightarrow$ CH$_{3}$OH + F$^{-}$',
                   'r6reverse':r'OH$^{-}$ + CH$_{3}$F $\leftarrow$ CH$_{3}$OH + F$^{-}$',
                   'r7forward':r'H$^{\bullet}$ + N$_{2}$ $\rightarrow$ N$_{2}$H$^{\bullet}$',
                   'r7reverse':r'H$^{\bullet}$ + N$_{2}$ $\leftarrow$ N$_{2}$H$^{\bullet}$',
                   'r8forward':r'H$^{\bullet}$ + C$_{2}$H$_{4}$ $\rightarrow$ CH$_{3}$CH$_{2}^{\bullet}$',
                   'r8reverse':r'H$^{\bullet}$ + C$_{2}$H$_{4}$ $\leftarrow$ CH$_{3}$CH$_{2}^{\bullet}$',
                   'r9forward':r'HCN $\rightarrow$ HNC',
                   'r9reverse':r'HCN $\leftarrow$ HNC',
                   'r10forward':r'OH$^{\bullet}$ + CH$_{4}$ $\rightarrow$ H$_{2}$O + CH$_{3}^{\bullet}$',
                   'r10reverse':r'OH$^{\bullet}$ + CH$_{4}$ $\leftarrow$ H$_{2}$O + CH$_{3}^{\bullet}$',
                   'r11forward':r'H$^{\bullet}$ + OH$^{\bullet}$ $\rightarrow$ O$^{\bullet \bullet}$ + H$_{2}$',
                   'r11reverse':r'H$^{\bullet}$ + OH$^{\bullet}$ $\leftarrow$ O$^{\bullet \bullet}$ + H$_{2}$',
                   'r12forward':r'H$^{\bullet}$ + H$_{2}$S $\rightarrow$ H$_{2}$ + HS$^{\bullet}$',
                   'r12reverse':r'H$^{\bullet}$ + H$_{2}$S $\leftarrow$ H$_{2}$ + HS$^{\bullet}$'}
    return dbh24_names[sys]


def rename_re42(sys):
    # Rename the indetifiers of RE42 into proper chemical notation
    re42_names = {'1-4-cyclohexadiene_2H2__cyclohexane':r'1-4-cyclo-C$_{6}$H$_{8}$+2H$_{2}$ $\rightarrow$ cyclo-C$_{6}$H$_{12}$',
                  'CH4_NH3__HCN_3H2':r'CH$_{4}$+NH$_{3}$ $\rightarrow$ HCN+3H$_{2}$',
                  'CO2_3H2__methanol_H2O':r'CO$_{2}$+3H$_{2}$ $\rightarrow$ CH$_{3}$OH+H$_{2}$O',
                  'CH4_2Cl2__CCl4_2H2':r'CH$_{4}$+2Cl$_{2}$ $\rightarrow$ CCl$_{4}$+2H$_{2}$',
                  'O2_H2__2OH':r'O$_{2}^{\bullet \bullet}$+H$_{2}$ $\rightarrow$ 2OH$^{\bullet}$',
                  'CH4_CO2__2CO_2H2':r'CH$_{4}$+CO$_{2}$ $\rightarrow$ 2CO+2H$_{2}$',
                  'CO_H2O__CO2_H2':r'CO+H$_{2}$O $\rightarrow$ CO$_{2}$+H$_{2}$',
                  'CH4_H2O__methanol_H2':r'CH$_{4}$+H$_{2}$O $\rightarrow$ CH$_{3}$OH+H$_{2}$',
                  'CO_2H2__methanol':r'CO+2H$_{2}$ $\rightarrow$ CH$_{3}$OH',
                  'oxirane_H2__ethene_H2O':r'CH$_{2}$OCH$_{2}$ + H$_{2}$ $\rightarrow$ C$_{2}$H$_{4}$+H$_{2}$O',
                  'CO_3H2__CH4_H2O':r'CO+3H$_{2}$ $\rightarrow$ CH$_{4}$+H$_{2}$O',
                  '2N2_O2__2N2O':r'2N$_{2}$+O$_{2}^{\bullet \bullet}$ $\rightarrow$ 2N$_{2}$O$^{\bullet}$',
                  'O2_2H2__2H2O':r'O$_{2}^{\bullet \bullet}$+2H$_{2}$ $\rightarrow$ 2H$_{2}$O',
                  'CO2_4H2__CH4_2H2O':r'CO$_{2}$+4H$_{2}$ $\rightarrow$ CH$_{4}$+2H$_{2}$O',
                  '1-3-cyclohexadiene__1-4-cyclohexadiene':r'1-4-cyclo-C$_{6}$H$_{8}$ $\rightarrow$ 1-3-cyclo-C$_{6}$H$_{8}$',
                  '2CO_O2__2CO2':r'2CO+O$_{2}^{\bullet \bullet}$ $\rightarrow$ 2CO$_{2}$',
                  'benzene_H2__1-4-cyclohexadiene':r'C$_{6}$H$_{6}$+H$_{2}$ $\rightarrow$ 1-4-cyclo-C$_{6}$H$_{8}$',
                  'CH4_2F2__CF4_2H2':r'CH$_{4}$+2F$_{2}$ $\rightarrow$ CF$_{4}$+2H$_{2}$',
                  '3O2__2O3':r'3O$_{2}^{\bullet \bullet}$ $\rightarrow$ 2O$_{3}$',
                  'N2_2O2__2NO2':r'N$_{2}$+2O$_{2}^{\bullet \bullet}$ $\rightarrow$ 2NO$_{2}$',
                  'propyne_H2__propene':r'C$_{3}$H$_{4}$-C3v+H$_{2}$ $\rightarrow$ C$_{3}$H$_{6}$',
                  'N2_3H2__2NH3':r'N$_{2}$+3H$_{2}$ $\rightarrow$ 3NH$_{3}$',
                  'CH4_CO_H2__ethanol':r'CH$_{4}$+CO+H$_{2}$ $\rightarrow$ C$_{2}$H$_{5}$OH',
                  'methylamine_H2__CH4_NH3':r'CH$_{3}$NH$_{2}$+H$_{2}$ $\rightarrow$ CH$_{4}$+NH$_{3}$',
                  'N2_2H2__N2H4':r'N$_{2}$+2H$_{2}$ $\rightarrow$ N$_{2}$H$_{4}$',
                  'O2_4HCl__2Cl2_2H2O':r'O$_{2}^{\bullet \bullet}$+4HCl $\rightarrow$ 2Cl$_{2}$ 2H$_{2}$O',
                  'allene_2H2__propane':r'C$_{3}$H$_{4}$-D2d+2H$_{2}$ $\rightarrow$ C$_{3}$H$_{8}$',
                  'propene_H2__propane':r'C$_{3}$H$_{6}$+H$_{2}$ $\rightarrow$ C$_{3}$H$_{8}$',
                  'CO_H2O__formic-acid':r'CO+H$_{2}$O $\rightarrow$ HCOOH',
                  'ethanol__dimethylether':r'C$_{2}$H$_{5}$OH $\rightarrow$ CH$_{3}$OCH$_{3}$',
                  'ethyne_H2__ethene':r'C$_{2}$H$_{4}$+2H$_{2}$ $\rightarrow$ C$_{2}$H$_{4}$',
                  '4CO_9H2__transbutane_4H2O':r'CO+9H$_{2}$ $\rightarrow$ trans-C$_{4}$H$_{10}$+4H$_{2}$O',
                  'ketene_2H2__ethene_H2O':r'H$_{2}$CCO+H$_{2}$ $\rightarrow$ C$_{2}$H$_{4}$+H$_{2}$O',
                  'isobutane__transbutane':r'iso-C$_{4}$H$_{10}$ $\rightarrow$ trans-C$_{4}$H$_{10}$',
                  'N2_O2__2NO':r'N$_{2}$+O$_{2}^{\bullet \bullet}$ $\rightarrow$ 2NO$^{\bullet}$',
                  'CH4_CO2__acetic-acid':r'CH$_{4}$+CO$_{2}$ $\rightarrow$ CH$_{3}$COOH',
                  '2OH_H2__2H2O':r'2OH$^{\bullet}$+H$_{2}$ $\rightarrow$ 2H$_{2}$O',
                  'H2_O2__H2O2':r'H$_{2}$+O$_{2}^{\bullet \bullet}$ $\rightarrow$ H$_{2}$O$_{2}$',
                  'thioethanol_H2__H2S_ethane':r'CH$_{3}$CH$_{2}$SH+H$_{2}$ $\rightarrow$ H$_{2}$S+C$_{2}$H$_{6}$',
                  '2methanol_O2__2CO2_4H2':r'2CH$_{3}$OH+O$_{2}^{\bullet \bullet}$ $\rightarrow$ 2CO$_{2}$+4H$_{2}$',
                  '2CO_2NO__2CO2_N2':r'2CO+2NO$^{\bullet}$ $\rightarrow$ 2CO$_{2}$+N$_{2}$',
                  'SO2_3H2__H2S_2H2O':r'SO$_{2}$+3H$_{2}$ $\rightarrow$ H$_{2}$S+2H$_{2}$O'}
    return re42_names[sys]    

def rename_s66x8(sys):
    s66x8_names = {'AcNH2AcNH2':r'AcNH$_{2}$ $\cdots$ AcNH$_{2}$',
                   'AcNH2Uracil':r'AcNH$_{2}$ $\cdots$ Uracil',
                   'AcOHAcOH':r'AcOH  $\cdots$ AcOH',
                   'AcOHUracil':r'AcOH  $\cdots$ Uracil',
                   'BenzeneAcNH2NHpi':r'C$_{6}$H$_{6} \cdots$ AcNH$_{2}$',
                   'BenzeneAcOH':r' C$_{6}$H$_{6} \cdots$ AcOH ($\|$)',
                   'BenzeneAcOHOHpi':r'C$_{6}$H$_{6} \cdots$ AcOH ($\perp$)',
                   'BenzeneBenzeneTS':r'C$_{6}$H$_{6} \cdots$ C$_{6}$H$_{6}$ ($\|$)',
                   'BenzeneBenzenepipi':r'C$_{6}$H$_{6} \cdots$ C$_{6}$H$_{6}$ ($\perp$)',
                   'BenzeneCyclopentane':r'C$_{6}$H$_{6} \cdots$ Cyclo-C$_{5}$H$_{10}$',
                   'BenzeneEthene':r'C$_{6}$H$_{6} \cdots$ C$_{2}$H$_{4}$',
                   'BenzeneEthyneCHpi':r'C$_{6}$H$_{6} \cdots$ C$_{2}$H$_{2}$',
                   'BenzeneMeNH2NHpi':r'C$_{6}$H$_{6} \cdots$ CH$_{3}$NH$_{2}$',
                   'BenzeneMeOHOHpi':r'C$_{6}$H$_{6} \cdots$ CH$_{3}$OH',
                   'BenzeneNeopentane':r'C$_{6}$H$_{6} \cdots$ Neo-C$_{5}$H$_{12}$',
                   'BenzenePeptideNHpi':r'C$_{6}$H$_{6} \cdots$ Peptide',
                   'BenzenePyridineTS':r'C$_{6}$H$_{6} \cdots$ Pyridine ($\|$)',
                   'BenzenePyridinepipi':r'C$_{6}$H$_{6} \cdots$ Pyridine ($\perp$)',
                   'BenzeneUracilpipi':r'C$_{6}$H$_{6} \cdots$ Uracil',
                   'BenzeneWaterOHpi':r'C$_{6}$H$_{6} \cdots$ H$_{2}$O',
                   'CyclopentaneCyclopentane':r'Cyclo-C$_{5}$H$_{10}$ $\cdots$ Cyclo-C$_{5}$H$_{10}$',
                   'CyclopentaneNeopentane':r'Cyclo-C$_{5}$H$_{10}$ $\cdots$ Neo-C$_{5}$H$_{12}$',
                   'EthenePentane':r'C$_{2}$H$_{4} \cdots$ C$_{5}$H$_{12}$',
                   'EthyneAcOHOHpi':r'C$_{2}$H$_{2} \cdots$ AcOH',
                   'EthyneEthyneTS':r'C$_{2}$H$_{2} \cdots$ C$_{2}$H$_{2}$ ($\perp$)',
                   'EthynePentane':r'C$_{2}$H$_{2} \cdots$ C$_{5}$H$_{12}$',
                   'EthyneWaterCHO':r'C$_{2}$H$_{2} \cdots$ H$_{2}$O',
                   'MeNH2MeNH2':r'CH$_{3}$NH$_{2} \cdots$ CH$_{3}$NH$_{2}$',
                   'MeNH2MeOH':r'CH$_{3}$NH$_{2} \cdots$ CH$_{3}$OH',
                   'MeNH2Peptide':r'CH$_{3}$NH$_{2} \cdots$ Peptide',
                   'MeNH2Pyridine':r'CH$_{3}$NH$_{2} \cdots$ Pyridine',
                   'MeNH2Water':r'CH$_{3}$NH$_{2} \cdots$ H$_{2}$O',
                   'MeOHMeNH2':r'CH$_{3}$OH $\cdots$ CH$_{3}$NH$_{2}$',
                   'MeOHMeOH':r'CH$_{3}$OH $\cdots$ CH$_{3}$OH',
                   'MeOHPeptide':r'CH$_{3}$OH $\cdots$ Peptide',
                   'MeOHPyridine':r'CH$_{3}$OH $\cdots$ Pyridine',
                   'MeOHWater':r'CH$_{3}$OH $\cdots$ H$_{2}$O',
                   'NeopentaneNeopentane':r'Neo-C$_{5}$H$_{12}$ $\cdots$ Neo-C$_{5}$H$_{12}$',
                   'NeopentanePentane':r'Neo-C$_{5}$H$_{12}$ $\cdots$ C$_{5}$H$_{12}$',
                   'PentaneAcNH2':r'C$_{5}$H$_{12} \cdots$ AcNH$_{2}$',
                   'PentaneAcOH':r'C$_{5}$H$_{12} \cdots$ AcOH',
                   'PentanePentane':r'C$_{5}$H$_{12} \cdots$ C$_{5}$H$_{12}$',
                   'PeptideEthene':r'Peptide $\cdots$ C$_{2}$H$_{4}$',
                   'PeptideMeNH2':r'Peptide $\cdots$ CH$_{3}$NH$_{2}$',
                   'PeptideMeOH':r'Peptide $\cdots$ CH$_{3}$OH',
                   'PeptidePentane':r'Peptide $\cdots$ C$_{5}$H$_{12}$',
                   'PeptidePeptide':r'Peptide $\cdots$ Peptide',
                   'PeptideWater':r'Peptide $\cdots$ H$_{2}$O',
                   'PyridineEthene':r'Pyridine $\cdots$ C$_{2}$H$_{4}$',
                   'PyridineEthyne':r'Pyridine $\cdots$ C$_{2}$H$_{2}$',
                   'PyridinePyridineCHN':r'Pyridine $\cdots$ Pyridine (in-plane)',
                   'PyridinePyridineTS':r'Pyridine $\cdots$ Pyridine ($\|$)',
                   'PyridinePyridinepipi':r'Pyridine $\cdots$ Pyridine ($\perp$)',
                   'PyridineUracilpipi':r'Pyridine $\cdots$ Uracil',
                   'UracilCyclopentane':r'Uracil $\cdots$ Cyclopentane',
                   'UracilEthene':r'Uracil $\cdots$ C$_{2}$H$_{4}$',
                   'UracilEthyne':r'Uracil $\cdots$ C$_{2}$H$_{2}$',
                   'UracilNeopentane':r'Uracil $\cdots$ Neo-C$_{5}$H$_{12}$',
                   'UracilPentane':r'Uracil $\cdots$ C$_{5}$H$_{12}$',
                   'UracilUracilBP':r'Uracil $\cdots$ Uracil (in-plane)',
                   'UracilUracilpipi':r'Uracil $\cdots$ Uracil ($\|$)',
                   'WaterMeNH2':r'H$_{2}$O $\cdots$ CH$_{3}$NH$_{2}$',
                   'WaterMeOH':r'H$_{2}$O $\cdots$ CH$_{3}$OH',
                   'WaterPeptide':r'H$_{2}$O $\cdots$ Peptide',
                   'WaterPyridine':r'H$_{2}$O $\cdots$ Pyridine',
                   'WaterWater':r'H$_{2}$O $\cdots$ H$_{2}$O'}
    return s66x8_names[sys.split('_')[0]], float(sys.split('_')[-1])

def rename_ads41(sys):
    ads41_names = {'C6H6+Ag111->C6H6@Ag111':r'C$_{6}$H$_{6}$ + Ag111     $\rightarrow$ C$_{6}$H$_{6}$@Ag111',
                   'C6H6+Au111->C6H6@Au111':r'C$_{6}$H$_{6}$ + Au111     $\rightarrow$ C$_{6}$H$_{6}$@Au111',
                   'C6H6+Cu111->C6H6@Cu111':r'C$_{6}$H$_{6}$ + Cu111     $\rightarrow$ C$_{6}$H$_{6}$@Cu111',
                   'C6H6+Pt111->C6H6@Pt111':r'C$_{6}$H$_{6}$ + Pt111     $\rightarrow$ C$_{6}$H$_{6}$@Pt111',
                   'C2H4+Pt111->[CCH3+H]@Pt111':r'C$_{2}$H$_{4}$ + Pt111     $\rightarrow$ CCH$_{3}$@Pt111 + H@Pt111',
                   'C2H6+Pt111->C2H6@Pt111':r'C$_{2}$H$_{6}$ + Pt111     $\rightarrow$ C$_{2}$H$_{6}$@Pt111',
                   'C3H8+Pt111->C3H8@Pt111':r'C$_{3}$H$_{8}$ + Pt111     $\rightarrow$ C$_{3}$H$_{8}$@Pt111',
                   'C4H10+Pt111->C4H10@Pt111':r'C$_{4}$H$_{10}$ + Pt111    $\rightarrow$ C$_{4}$H$_{10}$@Pt111',
                   'CH2I2+Pt111->[CH+H+I+I]@Pt111':r'CH$_{2}$I$_{2}$ + Pt111    $\rightarrow$ CH@Pt111 + H@Pt111 + 2 I@Pt111',
                   'CH3I+Pt111->CH3I@Pt111':r'CH$_{3}$I + Pt111          $\rightarrow$ CH$_{3}$I@Pt111',
                   'CH3I+Pt111->[CH3+I]@Pt111':r'CH$_{3}$I + Pt111          $\rightarrow$ CH$_{3}$@Pt111 + I@Pt111',
                   'CH4+Pt111->CH4@Pt111':r'CH$_{4}$ + Pt111           $\rightarrow$ CH$_{4}$@Pt111',
                   'CO+Co001->CO@Co001':r'CO + Co001                 $\rightarrow$ CO@Co001',
                   'CO+Cu111->CO@Cu111':r'CO + Cu111                 $\rightarrow$ CO@Cu111',
                   'CO+Ir111->CO@Ir111':r'CO + Ir111                 $\rightarrow$ CO@Ir111',
                   'CO+Ni111->CO@Ni111':r'CO + Ni111                 $\rightarrow$ CO@Ni111',
                   'CO+Pd100->CO@Pd100':r'CO + Pd100                 $\rightarrow$ CO@Pd100',
                   'CO+Pd111->CO@Pd111':r'CO + Pd111                 $\rightarrow$ CO@Pd111',
                   'CO+Pt111->CO@Pt111':r'CO + Pt111                 $\rightarrow$ CO@Pt111',
                   'CO+Rh111->CO@Rh111':r'CO + Rh111                 $\rightarrow$ CO@Rh111',
                   'CO+Ru001->CO@Ru001':r'CO + Ru001                 $\rightarrow$ CO@Ru001',
                   'C6H10+Pt111->C6H10@Pt111':r'C$_{6}$H$_{10}$ + Pt111    $\rightarrow$ C$_{6}$H$_{10}$@Pt111',
                   'H2O+O@Pt111->H2O-OD@Pt111':r'D$_{2}$O + $\frac{1}{3}$[O@Pt111] $\rightarrow$ $\frac{2}{3}$[(D$_{2}$O $\cdots$ OD)@Pt111]',
                   'H2O+Pt111->H2O@Pt111':r'D$_{2}$O + Pt111           $\rightarrow$ D$_{2}$O@Pt111',
                   'H2+Ni100->[H+H]@Ni100':r'H$_{2}$ + Ni100            $\rightarrow$ 2 H@Ni100',
                   'H2+Ni111->[H+H]@Ni111':r'H$_{2}$ + Ni111            $\rightarrow$ 2 H@Ni111',
                   'H2+Pd111->[H+H]@Pd111':r'H$_{2}$ + Pd111            $\rightarrow$ 2 H@Pd111',
                   'H2+Pt111->[H+H]@Pt111':r'H$_{2}$ + Pt111            $\rightarrow$ 2 H@Pt111',
                   'H2+Rh111->[H+H]@Rh111':r'H$_{2}$ + Rh111            $\rightarrow$ 2 H@Rh111',
                   'I2+Pt111->[I+I]@Pt111':r'I$_{2}$ + Pt111            $\rightarrow$ 2 I@Pt111',
                   'CH3OH+Pt111->CH3OH@Pt111':r'CH$_{3}$OH + Pt111         $\rightarrow$ CH$_{3}$OH@Pt111',
                   'C10H8+Pt111->C10H8@Pt111':r'C$_{10}$H$_{8}$ + Pt111    $\rightarrow$ C$_{10}$H$_{8}$@Pt111',
                   'NH3+Cu100->NH3@Cu100':r'NH$_{3}$ + Cu100           $\rightarrow$ NH$_{3}$@Cu100',
                   'NO+Ni100->[N+O]@Ni100':r'NO$^{\bullet}$ + Ni100     $\rightarrow$ N@Ni100 + O@Ni100',
                   'NO+Pd100->NO@Pd100':r'NO$^{\bullet}$ + Pd100     $\rightarrow$ NO@Pd100',
                   'NO+Pd111->NO@Pd111':r'NO$^{\bullet}$ + Pd111     $\rightarrow$ NO@Pd111',
                   'NO+Pt111->NO@Pt111':r'NO$^{\bullet}$ + Pt111     $\rightarrow$ NO@Pt111',
                   'O2+Ni100->[O+O]@Ni100':r'O$_{2}^{\bullet \bullet}$ + Ni100            $\rightarrow$ 2 O@Ni100',
                   'O2+Ni111->[O+O]@Ni111':r'O$_{2}^{\bullet \bullet}$ + Ni111            $\rightarrow$ 2 O@Ni111',
                   'O2+Pt111->[O+O]@Pt111':r'O$_{2}^{\bullet \bullet}$ + Pt111            $\rightarrow$ 2 O@Pt111',
                   'O2+Rh100->[O+O]@Rh100':r'O$_{2}^{\bullet \bullet}$ + Rh100            $\rightarrow$ 2 O@Rh100'}
    return ads41_names[sys]    

def rename_w411(sys):
    names_w411 = {'acetaldehyde':r'H$_3$C$-$C(=O)H',
                  'acetic':r'CH$_3$C(=O)OH',
                  'alcl':r'AlCl',
                  'alcl3':r'AlCl$_3$',
                  'alf':'AlF',
                  'alf3':r'AlF$_3$',
                  'alh':r'AlH',
                  'alh3':r'AlH$_3$',
                  'allene':r'H$_{2}$C=C=CH$_{2}$',
                  'b2':r'B$_2$',
                  'b2h6':r'B$_2$H$_6$',
                  'be2':r'Be$_2$',
                  'becl2':r'BeCl$_2$',
                  'bef2':r'BeF$_2$',
                  'bf':r'BF',
                  'bf3':r'BF$_3$',
                  'bh':r'BH',
                  'bh3':r'BH$_3$',
                  'bhf2':r'HBF$_2$',
                  'bn':r'BN $^1\Sigma^+$',
                  'bn3pi':r'BN $^3\Pi$',
                  'c-hcoh':r'cis-H(HO)C$^{\bullet \bullet}$',
                  'c-hono':r'cis-HON=O',
                  'c-hooo':r'cis-HO$_{3}^{\bullet}$',
                  'c-n2h2':r'cis-H$_2$N$_2$',
                  'c2':r'C$_2$',
                  'c2h2':r'C$_2$H$_2$',
                  'c2h3f':r'C$_2$H$_3$F',
                  'c2h4':r'C$_2$H$_4$',
                  'c2h5f':r'C$_2$H$_5$F',
                  'c2h6':r'C$_2$H$_6$',
                  'cch':r'C$\equiv$C$^{\bullet}$',
                  'ccl2':r'Cl$_2$C$^{\bullet \bullet}$',
                  'cf':r'FC$^{\bullet}$',
                  'cf2':r'$^{\bullet \bullet}$CF$_2$',
                  'cf4':r'CF$_4$',
                  'ch':r'HC$^{\bullet}$',
                  'ch2-sing':r'H$_2$C$^{\bullet \bullet}$ ($^1A_1$)',
                  'ch2-trip':r'H$_2$C$^{\bullet \bullet}$ ($^3B_2$)',
                  'ch2c':r'H$_2$C=C$^{\bullet \bullet}$',
                  'ch2ch':r'H$_2$C=HC$^{\bullet}$',
                  'ch2f2':r'CH$_2$F$_2$',
                  'ch2nh':r'H$_2$C=NH',
                  'ch2nh2':r'H$_2$N$-$CH$_2^{\bullet}$',
                  'ch3':r'H$_3$C$^{\bullet}$',
                  'ch3f':r'CH$_3$F',
                  'ch3nh':r'H$_3$C$-$NH$^{\bullet}$',
                  'ch3nh2':r'CH$_3$NH$_2$',
                  'ch4':r'CH$_4$',
                  'cl2':r'Cl$_2$',
                  'cl2o':r'ClOCl',
                  'clcn':r'ClC$\equiv$N',
                  'clf':r'ClF',
                  'clo':r'ClO$^{\bullet}$',
                  'cloo':r'OOCl$^{\bullet}$',
                  'cn':r'N$\equiv$C$^{\bullet}$',
                  'co':r'CO',
                  'co2':r'CO$_2$',
                  'cs':r'CS',
                  'cs2':r'CS$_2$',
                  'dioxirane':r'H$_2$CO$_2$',
                  'ethanol':r'C$_2$H$_5$OH',
                  'f2':r'F$_2$',
                  'f2co':r'F$_2$C=O',
                  'f2o':r'FOF',
                  'fccf':r'FC$\equiv$CF',
                  'fo2':r'OOF$^{\bullet}$',
                  'foof':r'F$_2$O$_2$',
                  'formic':r'HC(=O)OH',
                  'glyoxal':r'H(O=)C$-$C(=O)H',
                  'h2':r'H$_2$',
                  'h2cn':r'H$_2$C=N$^{\bullet}$',
                  'h2co':r'H$_2$C=O',
                  'h2o':r'H$_2$O',
                  'h2s':r'H$_2$S',
                  'hccf':r'HC$\equiv$CF',
                  'hcl':r'HCl',
                  'hcn':r'HCN',
                  'hcnh':r'HN=CH$^{\bullet}$',
                  'hcno':r'HCNO',
                  'hco':r'H(O=)C$^{\bullet}$',
                  'hcof':r'HC(=O)F',
                  'hf':r'HF',
                  'hnc':r'HNC$^{\bullet \bullet}$',
                  'hnco':r'HNCO',
                  'hnnn':r'HN$_3$',
                  'hno':r'HN=O',
                  'hocl':r'HOCl',
                  'hocn':r'HOCN',
                  'hof':r'HOF',
                  'honc':r'HONC$^{\bullet \bullet}$',
                  'hoo':r'HOO$^{\bullet}$',
                  'hooh':r'H$_2$O$_2$',
                  'hs':r'HS$^{\bullet}$',
                  'ketene':r'H$_2$C=C=O',
                  'methanol':r'CH$_3$OH',
                  'n2':r'N$_2$',
                  'n2h':r'HN$\equiv$N$^{\bullet}$',
                  'n2h4':r'N$_2$H$_4$',
                  'n2o':r'N$\equiv$N-O',
                  'nccn':r'N$\equiv$C$-$C$\equiv$N',
                  'nh':r'NH$^{\bullet}$',
                  'nh2':r'H$_2$N$^{\bullet}$',
                  'nh2cl':r'H$_2$NCl',
                  'nh3':r'NH$_3$',
                  'no':r'ON$^{\bullet}$',
                  'no2':r'O$_2$N$^{\bullet}$',
                  'o2':r'O$_2^{\bullet \bullet}$',
                  'o3':r'O$_3$',
                  'oclo':r'O$_2$Cl$^{\bullet}$',
                  'ocs':r'OCS',
                  'of':r'FO$^{\bullet}$',
                  'oh':r'HO$^{\bullet}$',
                  'oxirane':r'C$_2$H$_4$O',
                  'oxirene':r'H$_2$C$_2$O',
                  'p2':r'P$_2$',
                  'p4':r'P$_4$',
                  'ph3':r'PH$_3$',
                  'propane':r'C$_3$H$_8$',
                  'propene':r'C$_3$H$_6$',
                  'propyne':r'C$_3$H$_4$',
                  's2':r'S$_2^{\bullet \bullet}$',
                  's2o':r'SSO',
                  's3':r'S$_3$',
                  's4-c2v':r'S$_4$',
                  'si2h6':r'Si$_2$H$_6$',
                  'sif':r'FSi$^{\bullet}$',
                  'sif4':r'SiF$_4$',
                  'sih':r'HSi$^{\bullet}$',
                  'sih3f':r'SiH$_3$F',
                  'sih4':r'SiH$_4$',
                  'sio':r'SiO',
                  'so':r'SO$^{\bullet \bullet}$',
                  'so2':r'SO$_2$',
                  'so3':r'SO$_3$',
                  'ssh':r'HSS$^{\bullet}$',
                  't-hcoh':r'trans-H(HO)C$^{\bullet \bullet}$',
                  't-hono':r'trans-HON=O',
                  't-hooo':r'trans-HO$_3^{\bullet}$',
                  't-n2h2':r'trans-N$_2$H$_2$'}
    return names_w411[sys]



def get_idx(functional):
    '''
        get_idx
        -------
        Get index to be used to scan the file 'all_values.dat' for values of a specific funtional
    '''
    all_funct = ['PBE','PBE-D3','MS2','SCAN','r2SCAN','SCAN-rVV10','MCML','MCML-rVV10','VCML-rVV10','REF']
    idx_return = 3 # starting to count at 3, as the first entry is the DataSet, the second the System and the third the unit
    for idx,func in enumerate(all_funct):
        if functional == func:
            idx_return += idx
    return idx_return


class values():
    '''
        Class to define all values per functional
    '''
    def __init__(self):
        self.dbh24      = []  
        self.dbh24Sys   = []  
        self.re42       = []  
        self.re42Sys    = []  
        self.s66x8      = []  
        self.s66x8Sys   = []  
        self.alat       = []  
        self.alatSys    = []  
        self.ecoh       = []  
        self.ecohSys    = []  
        self.bulk       = []  
        self.bulkSys    = []  
        self.physi      = []  
        self.physiSys   = []  
        self.chemi      = []  
        self.chemiSys   = []  
        self.w411       = []  
        self.w411Sys    = []  
        self.label = ''

    # call to get errors from lines of an input file
    def get_values(self,functional):
        '''
            get_values
            ----------
            Get all values for a given functional

            Input
            -----
            functional ... which functional to evaluate 
            Possible values:
                PBE         
                PBE-D3         
                MS2
                SCAN
                r2SCAN
                MCML
                SCAN-rVV10
                MCML-rVV10
                VCML-rVV10
                REF
        '''
        self.label = functional     # set the label.
        # if labels are long: shorten them here
        if functional == 'PBE':
            self.color = 'green'
        if functional == 'PBE-D3':
            self.color = 'darkgreen'
        if functional == 'MS2':
            self.color = 'orange'
        if functional == 'SCAN':
            self.color = 'red'
        if functional == 'r2SCAN':
            self.color = 'darkred'
            self.label = r'r$^2$SCAN'
        if functional == 'SCAN-rVV10':
            self.color = 'pink'
            self.label = 'SCAN-v'
        if functional == 'MCML':
            self.color = 'blue'
            self.label = 'MCML'
        if functional == 'MCML-rVV10':
            self.color = 'purple'
            self.label = 'MCML-v'
        if functional == 'VCML-rVV10':
            self.color = 'lightblue'
            self.label = 'VCML-v'
        idx = get_idx(functional)   # Get the index corresponding to the position of th evalues in file 'all_values.dat'

        ffile = open(os.path.join(os.path.dirname(__file__), 'all_values.dat'), 'r')
        lines = ffile.readlines()
        ffile.close()

        for l,ll in enumerate(lines):
            if l > 0:
                splitt = ll.split()
                if splitt[0] == 'DBH24':
                    # splitt[idx] is the value for the given functional, splitt[1] is the corresponding system
                    self.dbh24.append(float(splitt[idx])) 
                    name = rename_dbh24(splitt[1])
                    self.dbh24Sys.append(name)
                if splitt[0] == 'RE42':
                    self.re42.append(float(splitt[idx])) 
                    name = rename_re42(splitt[1])
                    self.re42Sys.append(name)
                if splitt[0] == 'S66x8':
                    self.s66x8.append(float(splitt[idx]))
                    name, dist = rename_s66x8(splitt[1])
                    self.s66x8Sys.append([name,dist])
                if splitt[0] == 'alat':
                    self.alat.append(float(splitt[idx])) 
                    self.alatSys.append(splitt[1])
                if splitt[0] == 'Ecoh':
                    self.ecoh.append(float(splitt[idx]))
                    self.ecohSys.append(splitt[1])
                if splitt[0] == 'bulk':
                    self.bulk.append(float(splitt[idx])) 
                    self.bulkSys.append(splitt[1])
                if splitt[0] == 'ADS41':
                    is_physisorption = check_physi_or_chemi(splitt[1])
                    if is_physisorption == True:
                        self.physi.append(float(splitt[idx]))
                        name = rename_ads41(splitt[1])
                        self.physiSys.append(name)
                    else:
                        self.chemi.append(float(splitt[idx])) 
                        name = rename_ads41(splitt[1])
                        self.chemiSys.append(name)
                if splitt[0] == 'W4-11':
                    self.w411.append(float(splitt[idx])) 
                    name = rename_w411(splitt[1])
                    self.w411Sys.append(name)

        self.dbh24      = numpy.array(self.dbh24)
        self.dbh24Sys   = numpy.array(self.dbh24Sys)
        self.re42       = numpy.array(self.re42)
        self.re42Sys    = numpy.array(self.re42Sys)
        self.s66x8      = numpy.array(self.s66x8)
        self.s66x8Sys   = numpy.array(self.s66x8Sys)
        self.alat       = numpy.array(self.alat)
        self.alatSys    = numpy.array(self.alatSys)
        self.ecoh       = numpy.array(self.ecoh)
        self.ecohSys    = numpy.array(self.ecohSys)
        self.bulk       = numpy.array(self.bulk)
        self.bulkSys    = numpy.array(self.bulkSys)
        self.physi      = numpy.array(self.physi)
        self.physiSys   = numpy.array(self.physiSys)
        self.chemi      = numpy.array(self.chemi)
        self.chemiSys   = numpy.array(self.chemiSys)
        self.w411       = numpy.array(self.w411)
        self.w411Sys    = numpy.array(self.w411Sys)


def calc_errors(all_calc,ref):
    '''
        calculate errors for all functionals provided in all_calc
        ref are all reference values

        Insanely ugly right now, but it works
    '''
    # Errors for all functional and all data sets
    labels_all = [] # labels for the functionals
    labels_set = ['DBH24',
                  'RE42',
                  'S66x8',
                  'W4-11',
                  r'$a_{\mathrm{lat}}$@SOL62',
                  r'$E_{\mathrm{coh}}$@SOL62',
                  r'$B$@SOL62',
                  r'$E_{\mathrm{ads}}^{\mathrm{phy}}$@ADS41',
                  r'$E_{\mathrm{ads}}^{\mathrm{che}}$@ADS41'] # define labels for each set. Ugly, but it'll work
    units_set  = ['eV','eV','eV','eV',r'$\mathrm{\AA}$','eV','GPa','eV','eV'] # define units
    me_all   = []
    mae_all  = []
    mape_all = []
    for calc in all_calc:
        labels_all.append(calc.label)
        # error per functional
        me_tmp   = []
        mae_tmp  = []
        mape_tmp = []
        dataset_calc = [calc.dbh24,calc.re42,calc.s66x8,calc.w411,calc.alat,calc.ecoh,calc.bulk,calc.physi,calc.chemi]
        dataset_ref  = [ref.dbh24, ref.re42, ref.s66x8, ref.w411, ref.alat, ref.ecoh, ref.bulk, ref.physi, ref.chemi]
        for cal,re in zip(dataset_calc,dataset_ref):
            # calculate the error for each data set
            me, mae, mape = calc_errors_perSet(cal,re)
            me_tmp.append(me)
            mae_tmp.append(mae)
            mape_tmp.append(mape)
        me_all.append(me_tmp)
        mae_all.append(mae_tmp)
        mape_all.append(mape_tmp)

    # Header
    string = ''
    for a in range(len(labels_all)):
        string += ' & {:>10s}'.format(labels_all[a])

    # Reorganize the data
    tmp_all_me = []
    tmp_all_mae = []
    tmp_all_mape = []
    for l,label in enumerate(labels_all):
        tmp_me = []
        tmp_mae = []
        tmp_mape = []
        for e,me in enumerate(me_all[0]):
            tmp_me.append(me_all[l][e])
            tmp_mae.append(mae_all[l][e])
            tmp_mape.append(mape_all[l][e])
        tmp_all_me.append(tmp_me)
        tmp_all_mae.append(tmp_mae)
        tmp_all_mape.append(tmp_mape)

    # Actually print the numbers, here: 
    # Mean error
    print('### MEAN ERROR ###')
    print("\\begin{table}[h]")
    print("    \centering")
    print("    \caption{Mean errors for all data sets used within this study. Functionals with '-v' employ the rVV10 methodology. Units are provided.}")
    print("    \hspace*{-4em}")
    print("    \\begin{tabular}{lc|rrrrrrrrr}")
    print('Data Set                                 & Unit                '+string+' \\\\\\hline')
    form = "{:40.40s} & {:20.20s}" + " & ${:8.3f}$" *(len(tmp_all_me))
    for t in range(len(labels_set)):
        bubu = []
        bubu.append(labels_set[t])
        bubu.append(units_set[t])
        for ff in range(len(tmp_all_me)):
            bubu.append(tmp_all_me[ff][t])
        print(form.format(*bubu)+' \\\\')
    print("    \end{tabular}")
    print("    \label{tab:compErrors_ME}")
    print("\end{table}")


    # Actually print the numbers, here: 
    # Mean absolute error
    print('### MEAN ABSOLUTE ERROR ###')
    print("\\begin{table}[h]")
    print("    \centering")
    print("    \caption{Mean absolute errors for all data sets used within this study. Functionals with '-v' employ the rVV10 methodology. Units are provided.}")
    print("    \hspace*{-4em}")
    print("    \\begin{tabular}{lc|rrrrrrrrr}")
    print('Data Set                                 & Unit                '+string+' \\\\\\hline')
    form = "{:40.40s} & {:20.20s}" + " & {:10.3f}" *(len(tmp_all_mae))
    for t in range(len(labels_set)):
        bubu = []
        bubu.append(labels_set[t])
        bubu.append(units_set[t])
        for ff in range(len(tmp_all_mae)):
            bubu.append(tmp_all_mae[ff][t])
        print(form.format(*bubu)+' \\\\')
    print("    \end{tabular}")
    print("    \label{tab:compErrors_MAE}")
    print("\end{table}")



    # Actually print the numbers, here:
    # Mean absolute percentage error
    print('### MEAN ABSOLUTE PERCENTAGE ERROR ###')
    print("\\begin{table}[h]")
    print("    \centering")
    print("    \caption{Mean absolute percentage errors for all data sets used within this study. Functionals with '-v' employ the rVV10 methodology. Units are \\%.}")
    print("    \hspace*{-4em}")
    print("    \\begin{tabular}{l|rrrrrrrrr}")
    print('Data Set                                 '+string+' \\\\\\hline')
    form = "{:40.40s} " + " & {:10.3f}" *(len(tmp_all_mape))
    for t in range(len(labels_set)):
        bubu = []
        bubu.append(labels_set[t])
        for ff in range(len(tmp_all_mape)):
            bubu.append(tmp_all_mape[ff][t])
        print(form.format(*bubu)+' \\\\')
    print("    \end{tabular}")
    print("    \label{tab:compErrors_MAPE}")
    print("\end{table}")




def calc_errors_perSet(calc,ref):
    '''
        Calculate errors for any data set.
        calc should be, e.g., pbe.dbh24 while ref needs to be the corresponding ref.dbh24
    '''
    # Mean error
    me = numpy.sum(calc-ref)/len(calc)
    # Mean absolute error
    mae = numpy.sum(numpy.abs(calc-ref))/len(calc)
    # MEan absolute percentage error
    mape = numpy.sum(numpy.abs(calc-ref)/numpy.abs(ref))/len(calc)*100.0
    return me, mae, mape


def make_table_values(all_calc,ref,tag):
    '''
        create table for data sets, including all values of all functionals + REF

        all_calc: All functionals that shall be printed
        ref     : Reference values
        tag     : Define which dataset to print
    '''
   
    #########
    # DBH24 #
    #########
    if tag == 'DBH24':
        # HEADER
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Barrier heights for the DBH24 data set. Functionals with '-v' employ the rVV10 methodology. The units are eV, and the reference values are provided in the last column. The identifiers are defined in \\tabref{DBH24_system}.}")
        print("\\hspace*{-4em}")
        bla = 'r'*(len(all_calc)+1)  # alignment for all values printed
        print("\\begin{tabular}{l|"+bla+"}")

        # MIDDLE PART
        # string to assign the header names (functional names)
        string = '' 
        for a in range(len(all_calc)):
            string += ' & {:>12s}'.format(all_calc[a].label)
        string += ' & {:>12s}'.format('REF')
        # format string; starting with the system name
        form   = "{:7d}" + " & ${:10.3f}$" *(len(all_calc)+1)

        print('{:7.7s}'.format('System ')+string+' \\\\\hline')
        for v,val in enumerate(ref.dbh24):
            # array to store all values; as many entries as functionals
            all_val = numpy.zeros(len(all_calc))
            sys = v+1 #ref.dbh24Sys[v]
            for a in range(len(all_calc)):
                # Assign value for any functional to the correct index (given by the order in all_calc)
                all_val[a] = all_calc[a].dbh24[v]
            # print all values
            print(form.format(sys,*all_val,ref.dbh24[v])+' \\\\')

        # FOOTER
        print("\\end{tabular}")
        print("\\label{tab:DBH24_all}")
        print("\\end{table}")



    ########
    # RE42 #
    ########
    if tag == 'RE42':
        # HEADER
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Reaction energies for the RE42 data set. Functionals with '-v' employ the rVV10 methodology. The units are eV, and the reference values are provided in the last column. The identifiers are defined in \\tabref{RE42_system}.}")
        print("\\hspace*{-4em}")
        bla = 'r'*(len(all_calc)+1)  # alignment for all values printed
        print("\\begin{tabular}{l|"+bla+"}")

        # MIDDLE PART
        # string to assign the header names (functional names)
        string = ''
        for a in range(len(all_calc)):
            string += ' & {:>12s}'.format(all_calc[a].label)
        string += ' & {:>12s}'.format('REF')
        # format string; starting with the system name
        form   = "{:7d}" + " & ${:10.3f}$" *(len(all_calc)+1)

        print('{:7.7s}'.format('System ')+string+' \\\\\hline')
        for v,val in enumerate(ref.re42):
            # array to store all values; as many entries as functionals
            all_val = numpy.zeros(len(all_calc))
            sys = v+1 #ref.re42Sys[v]
            for a in range(len(all_calc)):
                # Assign value for any functional to the correct index (given by the order in all_calc)
                all_val[a] = all_calc[a].re42[v]
            # print all values
            print(form.format(sys,*all_val,ref.re42[v])+' \\\\')

        # FOOTER
        print("\\end{tabular}")
        print("\\label{tab:RE42_all}")
        print("\\end{table}")



    #########
    # S66x8 #
    #########
    #
    # For S66x8: One table per functional. print all 8 values into this one table
    #
    if tag == 'S66x8':
        distances = [0.9,0.95,1.0,1.05,1.1,1.25,1.5,2.0]
        for calc in all_calc:
            # HEADER
            print("\\begin{table}[h]")
            print("\\centering")
            print("\\tiny")
            if calc.label != 'REF':
                print("\\caption{Interaction energies for the S66x8 data set using "+calc.label+". The units are eV. Distances in the first row are relative to the equilibrium distance. The identifiers are defined in \\tabref{S66x8_system}.}")
            else:
                print("\\caption{Reference interaction energies for the S66x8 data set. The units are eV. Distances in the first row are relative to the equilibrium distance. The identifiers are defined in \\tabref{S66x8_system}.}")
            bla = 'r'*8  # alignment for all values printed
            print("\\begin{tabular}{l|"+bla+"}")

            # MIDDLE PART
            string = ''
            for a in range(8):
                string += ' & {:10.3f}'.format(distances[a])

            # format string; starting with the system name
            form   = "{:7d}" + " & ${:10.3f}$" * 8

            print('{:7.7s}'.format('System ')+' & ${:10.3f}$ & ${:10.3f}$ & ${:10.3f}$ & ${:10.3f}$ & ${:10.3f}$ & ${:10.3f}$ & ${:10.3f}$ & ${:10.3f}$'.format(*distances)+' \\\\\hline')
            count = 0
            all_val = []
            for v,val in enumerate(ref.s66x8):
                count = count + 1
                # array to store all values; as many entries as functionals
                sys = int((v+1)/8) 
                all_val.append(calc.s66x8[v])
                if count == 8:
                    # print all values
                    print(form.format(sys,*all_val)+' \\\\')
                    all_val = []
                    count = 0

            # FOOTER
            print("\\end{tabular}")
            if calc.label != 'r$^2$SCAN':
                print("\\label{tab:S66x8_all_"+calc.label+"}")
            else:
                print("\\label{tab:S66x8_all_r2SCAN}")
            print("\\end{table}")


    ########
    # alat #
    ########
    if tag == 'alat':
        # HEADER
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\tiny")
        print("\\caption{Lattice constants in the SOL62 data set. Functionals with '-v' employ the rVV10 methodology. The units are \\AA, and the reference values are provided in the last column. The errors per functional are shown in Figs.~\\ref{fig:errors_pbe2} (PBE, PBE-D3), \\ref{fig:errors_ms2mcml2} (MS2, MCML), \\ref{fig:errors_scanr2scan2} (SCAN, r$^{2}$SCAN), and \\ref{fig:errors_vdw2} (all functional using rVV10).}")
        bla = 'r'*(len(all_calc)+1)  # alignment for all values printed
        print("\\begin{tabular}{l|"+bla+"}")

        # MIDDLE PART
        # string to assign the header names (functional names)
        string = ''
        for a in range(len(all_calc)):
            string += ' & {:>12s}'.format(all_calc[a].label)
        string += ' & {:>12s}'.format('REF')
        # format string; starting with the system name
        form   = "{:7.7s}" + " & ${:10.3f}$" *(len(all_calc)+1)

        print('{:7.7s}'.format('System ')+string+' \\\\\hline')
        for v,val in enumerate(ref.alat):
            # array to store all values; as many entries as functionals
            all_val = numpy.zeros(len(all_calc))
            sys = ref.alatSys[v] #   v+1 #ref.re42Sys[v]
            for a in range(len(all_calc)):
                # Assign value for any functional to the correct index (given by the order in all_calc)
                all_val[a] = all_calc[a].alat[v]
            # print all values
            print(form.format(sys,*all_val,ref.alat[v])+' \\\\')

        # FOOTER
        print("\\end{tabular}")
        print("\\label{tab:alat_all}")
        print("\\end{table}")


    ########
    # Ecoh #
    ########
    if tag == 'Ecoh':
        # HEADER
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\tiny")
        print("\\caption{Cohesive energies in the SOL62 data set. Functionals with '-v' employ the rVV10 methodology. The units are eV, and the reference values are provided in the last column. The errors per functional are shown in Figs.~\\ref{fig:errors_pbe2} (PBE, PBE-D3), \\ref{fig:errors_ms2mcml2} (MS2, MCML), \\ref{fig:errors_scanr2scan2} (SCAN, r$^{2}$SCAN), and \\ref{fig:errors_vdw2} (all functional using rVV10).}")
        bla = 'r'*(len(all_calc)+1)  # alignment for all values printed
        print("\\begin{tabular}{l|"+bla+"}")

        # MIDDLE PART
        # string to assign the header names (functional names)
        string = ''
        for a in range(len(all_calc)):
            string += ' & {:>12s}'.format(all_calc[a].label)
        string += ' & {:>12s}'.format('REF')
        # format string; starting with the system name
        form   = "{:7.7s}" + " & ${:10.3f}$" *(len(all_calc)+1)

        print('{:7.7s}'.format('System ')+string+' \\\\\hline')
        for v,val in enumerate(ref.ecoh):
            # array to store all values; as many entries as functionals
            all_val = numpy.zeros(len(all_calc))
            sys = ref.ecohSys[v] #   v+1 #ref.re42Sys[v]
            for a in range(len(all_calc)):
                # Assign value for any functional to the correct index (given by the order in all_calc)
                all_val[a] = all_calc[a].ecoh[v]
            # print all values
            print(form.format(sys,*all_val,ref.ecoh[v])+' \\\\')

        # FOOTER
        print("\\end{tabular}")
        print("\\label{tab:ecoh_all}")
        print("\\end{table}")


    ########
    # bulk #
    ########
    if tag == 'bulk':
        # HEADER
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\tiny")
        print("\\caption{Bulk moduli in the SOL62 data set. Functionals with '-v' employ the rVV10 methodology. The units are GPa, and the reference values are provided in the last column. The errors per functional are shown in Figs.~\\ref{fig:errors_pbe2} (PBE, PBE-D3), \\ref{fig:errors_ms2mcml2} (MS2, MCML), \\ref{fig:errors_scanr2scan2} (SCAN, r$^{2}$SCAN), and \\ref{fig:errors_vdw2} (all functional using rVV10).}")
        bla = 'r'*(len(all_calc)+1)  # alignment for all values printed
        print("\\begin{tabular}{l|"+bla+"}")

        # MIDDLE PART
        # string to assign the header names (functional names)
        string = ''
        for a in range(len(all_calc)):
            string += ' & {:>12s}'.format(all_calc[a].label)
        string += ' & {:>12s}'.format('REF')
        # format string; starting with the system name
        form   = "{:7.7s}" + " & ${:10.3f}$" *(len(all_calc)+1)

        print('{:7.7s}'.format('System ')+string+' \\\\\hline')
        for v,val in enumerate(ref.bulk):
            # array to store all values; as many entries as functionals
            all_val = numpy.zeros(len(all_calc))
            sys = ref.bulkSys[v] #   v+1 #ref.re42Sys[v]
            for a in range(len(all_calc)):
                # Assign value for any functional to the correct index (given by the order in all_calc)
                all_val[a] = all_calc[a].bulk[v]
            # print all values
            print(form.format(sys,*all_val,ref.bulk[v])+' \\\\')

        # FOOTER
        print("\\end{tabular}")
        print("\\label{tab:bulk_all}")
        print("\\end{table}")


    #########
    # ADS41 #
    #########
    if tag == 'ADS41':
        # HEADER
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Adsorption energies in the ADS41 data set. Functionals with '-v' employ the rVV10 methodology. The units are eV. Top: physisorption-dominated; bottom: chemisorption-dominated. The reference values are provided in the last column, while the identifiers are defined in \\tabref{ADS41_system}.}")
        print("\\hspace*{-4em}")
        bla = 'r'*(len(all_calc)+1)  # alignment for all values printed
        print("\\begin{tabular}{l|"+bla+"}")

        # MIDDLE PART
        # string to assign the header names (functional names)
        string = ''
        for a in range(len(all_calc)):
            string += ' & {:>12s}'.format(all_calc[a].label)
        string += ' & {:>12s}'.format('REF')
        # format string; starting with the system name
        form   = "{:7d}" + " & ${:10.3f}$" *(len(all_calc)+1)

        print('{:7.7s}'.format('System ')+string+' \\\\\hline')
        # Physisorption
        for v,val in enumerate(ref.physi):
            # array to store all values; as many entries as functionals
            all_val = numpy.zeros(len(all_calc))
            sys = v + 1 #ref.bulkSys[v] #   v+1 #ref.re42Sys[v]
            for a in range(len(all_calc)):
                # Assign value for any functional to the correct index (given by the order in all_calc)
                all_val[a] = all_calc[a].physi[v]
            # print all values
            print(form.format(sys,*all_val,ref.physi[v])+' \\\\')
        print('\\hline')
        # Chemisorption
        for v,val in enumerate(ref.chemi):
            # array to store all values; as many entries as functionals
            all_val = numpy.zeros(len(all_calc))
            sys = v + 1 #ref.bulkSys[v] #   v+1 #ref.re42Sys[v]
            for a in range(len(all_calc)):
                # Assign value for any functional to the correct index (given by the order in all_calc)
                all_val[a] = all_calc[a].chemi[v]
            # print all values
            print(form.format(sys,*all_val,ref.chemi[v])+' \\\\')

        # FOOTER
        print("\\end{tabular}")
        print("\\label{tab:ADS41_all}")
        print("\\end{table}")


    #########
    # W4-11 #
    #########
    if tag == 'W4-11':
        # HEADER
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\footnotesize")
        print("\\caption{Atomization energies for the W4-11 data set. Functionals with '-v' employ the rVV10 methodology. The units are eV, and the reference values are provided in the last column. The identifiers are defined in \\tabref{W411_system}.}")
        bla = 'r'*(len(all_calc)+1)  # alignment for all values printed
        print("\\begin{tabular}{l|"+bla+"}")

        # MIDDLE PART
        # string to assign the header names (functional names)
        string = ''
        for a in range(len(all_calc)):
            string += ' & {:>12s}'.format(all_calc[a].label)
        string += ' & {:>12s}'.format('REF')
        # format string; starting with the system name
        form   = "{:7d}" + " & ${:10.3f}$" *(len(all_calc)+1)

        print('{:7.7s}'.format('System ')+string+' \\\\\hline')
        counter = 0
        for v,val in enumerate(ref.w411):
            counter += 1
            # array to store all values; as many entries as functionals
            all_val = numpy.zeros(len(all_calc))
            sys = v+1 #ref.re42Sys[v]
            for a in range(len(all_calc)):
                # Assign value for any functional to the correct index (given by the order in all_calc)
                all_val[a] = all_calc[a].w411[v]
            # print all values
            print(form.format(sys,*all_val,ref.w411[v])+' \\\\')

            # need to break the table, because too many entries
            if counter in [47,94]:
                print("\\end{tabular}")
                print("\\end{table}")
                print("\\addtocounter{table}{-1}")
                print("\\begin{table}[h]")
                print("\\centering")
                print("\\footnotesize")
                print("\\caption{Atomization energies for the W4-11 data set. Continued.}")
                bla = 'r'*(len(all_calc)+1)  # alignment for all values printed
                print("\\begin{tabular}{l|"+bla+"}")
                print('{:7.7s}'.format('System ')+string+' \\\\\hline')

        # FOOTER
        print("\\end{tabular}")
        print("\\label{tab:W411_all}")
        print("\\end{table}")




def make_table_system(ref,tag):
    '''
        create table, including all systems

        ref     : Reference values
        tag     : Define which dataset to print
    '''

    #########
    # DBH24 #
    #########
    if tag == 'DBH24':
        # HEADER
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Systems in the DBH24 data set. Identifiers are provided, which are used in \\tabref{DBH24_all} and Figs.~\\ref{fig:errors_pbe} (PBE, PBE-D3), \\ref{fig:errors_ms2mcml} (MS2, MCML), \\ref{fig:errors_scanr2scan} (SCAN, r$^{2}$SCAN), and \\ref{fig:errors_vdw} (all functional using rVV10).}")
        print("\\begin{tabular}{l|l}")
        # MIDDLE PART
        print('{:11.11s}'.format('Identifier ')+'& System \\\\\hline')
        for v,val in enumerate(ref.dbh24):
            # array to store all values; as many entries as functionals
            idx = v + 1
            sys = ref.dbh24Sys[v]
            print('{:10d} & {}'.format(idx,sys)+' \\\\')
        # FOOTER
        print("\\end{tabular}")
        print("\\label{tab:DBH24_system}")
        print("\\end{table}")

    ########
    # RE42 #
    ########
    if tag == 'RE42':
        # HEADER
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Systems in the RE42 data set. Identifiers are provided, which are used in \\tabref{RE42_all} and Figs.~\\ref{fig:errors_pbe} (PBE, PBE-D3), \\ref{fig:errors_ms2mcml} (MS2, MCML), \\ref{fig:errors_scanr2scan} (SCAN, r$^{2}$SCAN), and \\ref{fig:errors_vdw} (all functionals using rVV10).}")
        print("\\begin{tabular}{l|l}")
        # MIDDLE PART
        print('{:11.11s}'.format('Identifier ')+'& System \\\\\hline')
        for v,val in enumerate(ref.re42):
            # array to store all values; as many entries as functionals
            idx = v + 1
            sys = ref.re42Sys[v]
            print('{:10d} & {}'.format(idx,sys)+' \\\\')
        # FOOTER
        print("\\end{tabular}")
        print("\\label{tab:RE42_system}")
        print("\\end{table}")

    #########
    # S66x8 #
    #########
    if tag == 'S66x8':
        # HEADER
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\tiny")
        print("\\caption{Systems in the S66x8 data set. Identifiers are provided, which are used in Tabs.~\\ref{tab:S66x8_all_PBE} (PBE), \\ref{tab:S66x8_all_PBE-D3} (PBE-D3), \\ref{tab:S66x8_all_MS2} (MS2), \\ref{tab:S66x8_all_SCAN} (SCAN), \\ref{tab:S66x8_all_r2SCAN} (r$^{2}$SCAN),  \\ref{tab:S66x8_all_SCAN-v} (SCAN-v), \\ref{tab:S66x8_all_MCML} (MCML), \\ref{tab:S66x8_all_MCML-v} (MCML-v), \\ref{tab:S66x8_all_VCML-v} (VCML-v), and \\ref{tab:S66x8_all_REF} (Reference values) as well as Figs.~\\ref{fig:errors_pbe} (PBE, PBE-D3), \\ref{fig:errors_ms2mcml} (MS2, MCML), \\ref{fig:errors_scanr2scan} (SCAN, r$^{2}$SCAN), and \\ref{fig:errors_vdw} (all functionals using rVV10).}")
        print("\\begin{tabular}{l|l}")
        # MIDDLE PART
        print('{:11.11s}'.format('Identifier ')+'& System \\\\\hline')
        count = 0
        for v,val in enumerate(ref.s66x8):
            count += 1
            # array to store all values; as many entries as functionals
            idx = int((v + 1)/8)
            sys = ref.s66x8Sys[v][0]
            if count == 8:
                print('{:10d} & {}'.format(idx,sys)+' \\\\')
                count = 0
        # FOOTER
        print("\\end{tabular}")
        print("\\label{tab:S66x8_system}")
        print("\\end{table}")


    #########
    # SOL62 #
    #########
    if tag == 'SOL62':
        # HEADER
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\tiny")
        print("\\caption{Systems in the SOL62 data set.}")
        print("\\begin{tabular}{l|l}")
        # MIDDLE PART
        print('{:11.11s}'.format('Identifier ')+'& System \\\\\hline')
        for v,val in enumerate(ref.alat):
            # array to store all values; as many entries as functionals
            idx = v+1
            sys = ref.alatSys[v]
            print('{:10d} & {:10.10s}'.format(idx,sys)+' \\\\')
        # FOOTER
        print("\\end{tabular}")
        print("\\label{tab:SOL62_system}")
        print("\\end{table}")


    #########
    # ADS41 #
    #########
    if tag == 'ADS41':
        # HEADER
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{Systems in the ADS41 data set. Identifiers are provided, which are used in \\tabref{ADS41_all} and Figs.~\\ref{fig:errors_pbe3} (PBE, PBE-D3), \\ref{fig:errors_ms2mcml3} (MS2, MCML), \\ref{fig:errors_scanr2scan3} (SCAN, r$^{2}$SCAN), and \\ref{fig:errors_vdw3} (all functionals using rVV10). Top: physisorption-dominated; bottom: chemisorption-dominated.}")
        print("\\begin{tabular}{l|l}")
        # MIDDLE PART
        print('{:11.11s}'.format('Identifier ')+'& System \\\\\hline')
        for v,val in enumerate(ref.physi):
            # array to store all values; as many entries as functionals
            idx = v + 1
            sys = ref.physiSys[v]
            print('{:10d} & {}'.format(idx,sys)+' \\\\')
        print('\\hline')
        for v,val in enumerate(ref.chemi):
            # array to store all values; as many entries as functionals
            idx = v + 1
            sys = ref.chemiSys[v]
            print('{:10d} & {}'.format(idx,sys)+' \\\\')
        # FOOTER
        print("\\end{tabular}")
        print("\\label{tab:ADS41_system}")
        print("\\end{table}")

    #########
    # W4-11 #
    #########
    if tag == 'W4-11':
        # HEADER
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\footnotesize")
        print("\\caption{Systems in the W4-11 data set. Identifiers are provided, which are used in \\tabref{W411_all} and Figs.~\\ref{fig:errors_pbe3} (PBE, PBE-D3), \\ref{fig:errors_ms2mcml3} (MS2, MCML), \\ref{fig:errors_scanr2scan3} (SCAN, r$^{2}$SCAN), and \\ref{fig:errors_vdw3} (all functionals using rVV10).}")
        print("\\begin{tabular}{l|l}")
        # MIDDLE PART
        print('{:11.11s}'.format('Identifier ')+'& System \\\\\hline')
        counter = 0
        for v,val in enumerate(ref.w411):
            counter += 1
            # array to store all values; as many entries as functionals
            idx = v+1
            sys = ref.w411Sys[v]
            print('{:10d} & {}'.format(idx,sys)+' \\\\')

            # break table, as too long
            if counter in [47,94]:
                print("\\end{tabular}")
                print("\\end{table}")
                print("\\addtocounter{table}{-1}")
                print("\\begin{table}[h]")
                print("\\centering")
                print("\\footnotesize")
                print("\\caption{Systems in the W4-11 data set. Continued.}")
                print("\\begin{tabular}{l|l}")
                print('{:11.11s}'.format('Identifier ')+'& System \\\\\hline')
        # FOOTER
        print("\\end{tabular}")
        print("\\label{tab:W411_system}")
        print("\\end{table}")



def plot_errors(functs,ref):
    '''
        Plot the errors for all functionals 
        Plot each dataset individually
    '''
    lw = 1.0
    fz = 25
    dpi_value = 300

    # DBH24
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:5.2f}'))
    all_names = ''
    for funct in functs:
        # change abbreviated rVV10 functionals back
        if funct.label[-1] == 'v':
            name = funct.label.split('-')[0]+'-rVV10'
        else:
            name = funct.label
        # make sure the final figure is labelled correctly
        if name == 'r$^2$SCAN':
            all_names += 'r2SCAN'
        elif name[-4:] == 'VV10':
            all_names = 'vdW'
        else:
            all_names += name
        # plot data
        plt.plot(funct.dbh24Sys,funct.dbh24-ref.dbh24,'-o',color=funct.color,linewidth=lw,label=name)
    plt.plot([0,len(funct.dbh24)-1],[0.0,0.0],'-',color='black',linewidth=1.0)
    plt.legend(fontsize=fz)
    plt.xticks(funct.dbh24Sys,numpy.arange(1,len(funct.dbh24)+1),fontsize=fz,rotation=90)
    plt.yticks(fontsize=fz)
    plt.xlim([-0.1,len(funct.dbh24Sys)-1+0.1])
    plt.ylim([-1.35,0.15])
    plt.ylabel('Error, DBH24 [eV]',fontsize=fz)
    plt.savefig('pics/DBH24_errors_{}.png'.format(all_names), bbox_inches='tight',pad_inches = 0, dpi=dpi_value)
    #plt.show()
    
    # RE42
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:5.2f}'))
    all_names = ''
    for funct in functs:
        # change abbreviated rVV10 functionals back
        if funct.label[-1] == 'v':
            name = funct.label.split('-')[0]+'-rVV10'
        else:
            name = funct.label
        # make sure the final figure is labelled correctly
        if name == 'r$^2$SCAN':
            all_names += 'r2SCAN'
        elif name[-4:] == 'VV10':
            all_names = 'vdW'
        else:
            all_names += name
        # plot data
        plt.plot(funct.re42Sys,funct.re42-ref.re42,'-o',color=funct.color,linewidth=lw,label=name)
    plt.plot([0,len(funct.re42)-1],[0.0,0.0],'-',color='black',linewidth=1.0)
    plt.legend(fontsize=fz)
    plt.xticks(funct.re42Sys,numpy.arange(1,len(funct.re42)+1),fontsize=fz,rotation=90)
    plt.yticks(fontsize=fz)
    plt.xlim([-0.1,len(funct.re42Sys)-1+0.1])
    plt.ylim([-1.55,1.60])
    plt.ylabel('Error, RE42 [eV]',fontsize=fz)
    plt.savefig('pics/RE42_errors_{}.png'.format(all_names), bbox_inches='tight',pad_inches = 0, dpi=dpi_value)
    #plt.show()
    
    # S66x8
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:5.2f}'))
    all_names = ''
    for funct in functs:
        # change abbreviated rVV10 functionals back
        if funct.label[-1] == 'v':
            name = funct.label.split('-')[0]+'-rVV10'
        else:
            name = funct.label
        # make sure the final figure is labelled correctly
        if name == 'r$^2$SCAN':
            all_names += 'r2SCAN'
        elif name[-4:] == 'VV10':
            all_names = 'vdW'
        else:
            all_names += name
        # plot data
        plt.plot(funct.s66x8-ref.s66x8,'-o',color=funct.color,linewidth=lw,label=name)
    plt.plot([0,len(funct.s66x8)-1],[0.0,0.0],'-',color='black',linewidth=1.0)
    plt.legend(fontsize=fz)
    # Shift x lable position by 4.5, so that the name is in the middle of the 8 data points
    system  = []
    counter = []
    count = -4.5
    for name in funct.s66x8Sys[:,0]:
        sys = name
        if sys not in system:
            system.append(sys)
            count += 8
            counter.append(count)
    plt.xticks(counter,numpy.arange(1,len(system)+1),fontsize=15,rotation=90)
    plt.yticks(fontsize=fz)
    plt.xlim([-0.1,len(funct.s66x8Sys)-1+0.1])
    plt.ylim([-0.22,0.45])
    plt.ylabel('Error, S66x8 [eV]',fontsize=fz)
    plt.savefig('pics/S66x8_errors_{}.png'.format(all_names), bbox_inches='tight',pad_inches = 0, dpi=dpi_value)
    #plt.show()
    
    # W4-11
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:5.2f}'))
    all_names = ''
    for funct in functs:
        # change abbreviated rVV10 functionals back
        if funct.label[-1] == 'v':
            name = funct.label.split('-')[0]+'-rVV10'
        else:
            name = funct.label
        # make sure the final figure is labelled correctly
        if name == 'r$^2$SCAN':
            all_names += 'r2SCAN'
        elif name[-4:] == 'VV10':
            all_names = 'vdW'
        else:
            all_names += name
        # plot data
        plt.plot(funct.w411Sys,funct.w411-ref.w411,'-o',color=funct.color,linewidth=lw,label=name)
    plt.plot([0,len(funct.w411)-1],[0.0,0.0],'-',color='black',linewidth=1.0)
    plt.legend(fontsize=fz)
    plt.xticks(funct.w411Sys,numpy.arange(1,len(funct.w411)+1),fontsize=8,rotation=90)
    plt.yticks(fontsize=fz)
    plt.xlim([-0.1,len(funct.w411Sys)-1+0.1])
    plt.ylim([-2.80,2.25])
    plt.ylabel('Error, W4-11 [eV]',fontsize=fz)
    plt.savefig('pics/W411_errors_{}.png'.format(all_names), bbox_inches='tight',pad_inches = 0, dpi=dpi_value)
    #plt.show()
    
    # alat
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:5.2f}'))
    all_names = ''
    for funct in functs:
        # change abbreviated rVV10 functionals back
        if funct.label[-1] == 'v':
            name = funct.label.split('-')[0]+'-rVV10'
        else:
            name = funct.label
        # make sure the final figure is labelled correctly
        if name == 'r$^2$SCAN':
            all_names += 'r2SCAN'
        elif name[-4:] == 'VV10':
            all_names = 'vdW'
        else:
            all_names += name
        # plot data
        plt.plot(funct.alatSys,funct.alat-ref.alat,'-o',color=funct.color,linewidth=lw,label=name)
    plt.plot([0,len(funct.alat)-1],[0.0,0.0],'-',color='black',linewidth=1.0)
    plt.legend(fontsize=fz)
    plt.xticks(fontsize=15,rotation=90)
    plt.yticks(fontsize=fz)
    plt.xlim([-0.1,len(funct.alatSys)-1+0.1])
    plt.ylim([-0.13,0.195])
    plt.ylabel(r'Error, $a_{\mathrm{lat}}$@SOL62 [$\mathrm{\AA}$]',fontsize=fz)
    plt.savefig('pics/alat_errors_{}.png'.format(all_names), bbox_inches='tight',pad_inches = 0, dpi=dpi_value)
    #plt.show()
    
    # ecoh
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:5.2f}'))
    all_names = ''
    for funct in functs:
        # change abbreviated rVV10 functionals back
        if funct.label[-1] == 'v':
            name = funct.label.split('-')[0]+'-rVV10'
        else:
            name = funct.label
        # make sure the final figure is labelled correctly
        if name == 'r$^2$SCAN':
            all_names += 'r2SCAN'
        elif name[-4:] == 'VV10':
            all_names = 'vdW'
        else:
            all_names += name
        # plot data
        plt.plot(funct.ecohSys,funct.ecoh-ref.ecoh,'-o',color=funct.color,linewidth=lw,label=name)
    plt.plot([0,len(funct.ecoh)-1],[0.0,0.0],'-',color='black',linewidth=1.0)
    plt.legend(fontsize=fz)
    plt.xticks(fontsize=15,rotation=90)
    plt.yticks(fontsize=fz)
    plt.xlim([-0.1,len(funct.ecohSys)-1+0.1])
    plt.ylim([-1.32,1.27])
    plt.ylabel(r'Error, $E_{\mathrm{coh}}$@SOL62 [eV]',fontsize=fz)
    plt.savefig('pics/ecoh_errors_{}.png'.format(all_names), bbox_inches='tight',pad_inches = 0, dpi=dpi_value)
    #plt.show()
    
    # bulk
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:5.1f}'))
    all_names = ''
    for funct in functs:
        # change abbreviated rVV10 functionals back
        if funct.label[-1] == 'v':
            name = funct.label.split('-')[0]+'-rVV10'
        else:
            name = funct.label
        # make sure the final figure is labelled correctly
        if name == 'r$^2$SCAN':
            all_names += 'r2SCAN'
        elif name[-4:] == 'VV10':
            all_names = 'vdW'
        else:
            all_names += name
        # plot data
        plt.plot(funct.bulkSys,funct.bulk-ref.bulk,'-o',color=funct.color,linewidth=lw,label=name)
    plt.plot([0,len(funct.bulk)-1],[0.0,0.0],'-',color='black',linewidth=1.0)
    plt.legend(fontsize=fz)
    plt.xticks(fontsize=15,rotation=90)
    plt.yticks(fontsize=fz)
    plt.xlim([-0.1,len(funct.bulkSys)-1+0.1])
    plt.ylim([-45,76])
    plt.ylabel(r'Error, $B$@SOL62 [eV]',fontsize=fz)
    plt.savefig('pics/bulk_errors_{}.png'.format(all_names), bbox_inches='tight',pad_inches = 0, dpi=dpi_value)
    #plt.show()
    
    # ADS41
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:5.2f}'))
    all_names = ''
    for funct in functs:
        # change abbreviated rVV10 functionals back
        if funct.label[-1] == 'v':
            name = funct.label.split('-')[0]+'-rVV10'
        else:
            name = funct.label
        # make sure the final figure is labelled correctly
        if name == 'r$^2$SCAN':
            all_names += 'r2SCAN'
        elif name[-4:] == 'VV10':
            all_names = 'vdW'
        else:
            all_names += name
        # plot data
        names = funct.physiSys.tolist() + funct.chemiSys.tolist()
        error = (funct.physi-ref.physi).tolist() + (funct.chemi-ref.chemi).tolist()
        plt.plot(names,error,'-o',color=funct.color,linewidth=lw,label=name)
    plt.plot([0,len(names)-1],[0.0,0.0],'-',color='black',linewidth=1.0)
    # Separate Physi from chemi
    plt.plot([len(funct.physi)-0.5,len(funct.physi)-0.5],[-10.0,10.0],color='black',linewidth=1.0)
    plt.legend(fontsize=fz)
    plt.xticks(names,numpy.arange(1,len(funct.physi)+1).tolist()+numpy.arange(1,len(funct.chemi)+1).tolist(),fontsize=fz,rotation=90)
    plt.yticks(fontsize=fz)
    plt.xlim([-0.1,len(funct.physiSys)+len(funct.chemiSys)-1+0.1])
    plt.ylim([-1.20,1.25])
    plt.ylabel(r'Error, $E_{\mathrm{ads}}^{\mathrm{phy/che}}$@ADS41 [eV]',fontsize=fz)
    plt.savefig('pics/ADS41_errors_{}.png'.format(all_names), bbox_inches='tight',pad_inches = 0, dpi=dpi_value)
    #plt.show()
    plt.close('all')





def plot_barPlots(functs,ref,tag='MAE'):
    '''
        Make bar plots for ME, MAE, or MAPE
    '''
    datasets    = ['DBH24\n(eV)',
                   'RE42\n(eV)',
                   'S66x8\n(eV)',
                   'W4-11\n(eV)',
                   r'$a_{\mathrm{lat}}$@SOL62'+'\n($\mathrm{\AA}$)',
                   r'$E_{\mathrm{coh}}$@SOL62'+'\n(eV)',
                   r'$B$@SOL62'+'\n(100 GPa)',
                   r'$E_{\mathrm{ads}}^{\mathrm{phy}}$'+'@ADS41\n(eV)',
                   r'$E_{\mathrm{ads}}^{\mathrm{che}}$'+'@ADS41\n(eV)']
    barWidth    = 0.20
    fz          = 30
    fig = plt.subplots(figsize = (16, 9))
    # heights of bars -> data set numbers
    # Set position of bar on X axis
    br1 = numpy.arange(len(datasets))*1.95 # stretch x spacing for plot
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]
    br7 = [x + barWidth for x in br6]
    br8 = [x + barWidth for x in br7]
    br9 = [x + barWidth for x in br8]
    all_br = [br1,br2,br3,br4,br5,br6,br7,br8,br9]
    #
    # Make plt 
    # Get ME, MAE for any functional
    for c,calc in enumerate(functs):
        all_me  = []
        all_mae = []
        all_mape = []
        dataset_calc = [calc.dbh24,calc.re42,calc.s66x8,calc.w411,calc.alat,calc.ecoh,calc.bulk,calc.physi,calc.chemi]
        dataset_ref  = [ref.dbh24, ref.re42, ref.s66x8, ref.w411, ref.alat, ref.ecoh, ref.bulk, ref.physi, ref.chemi]
        for index, (cal,re) in enumerate(zip(dataset_calc,dataset_ref)):
            # calculate the error for each data set
            me, mae, mape = calc_errors_perSet(cal,re)
            # adjust error for bulk moduli
            if index == 6:
                me  /= 100.0
                mae /= 100.0
            all_me.append(me)
            all_mae.append(mae)
            all_mape.append(mape)
        # change abbreviated rVV10 functionals back
        if calc.label[-1] == 'v':
            name = calc.label.split('-')[0]+'-rVV10'
        else:
            name = calc.label
        # Plot the ME or MAE
        if tag == 'MAE':
            plt.bar(all_br[c], all_mae,  color = calc.color, width = barWidth, edgecolor = 'grey', label = name)
        if tag == 'ME':
            plt.bar(all_br[c], all_me,   color = calc.color, width = barWidth, edgecolor = 'grey', label = name)
        if tag == 'MAPE':
            plt.bar(all_br[c], all_mape, color = calc.color, width = barWidth, edgecolor = 'grey', label = name)
    # Adding Xticks
    if tag == 'MAPE':
        plt.ylabel('MAPE [%]', fontsize=fz)
        datasets = ['DBH24',
                    'RE42',
                    'S66x8',
                    'W4-11',
                    r'$a_{\mathrm{lat}}$@SOL62',
                    r'$E_{\mathrm{coh}}$@SOL62',
                    r'$B$@SOL62',
                    r'$E_{\mathrm{ads}}^{\mathrm{phy}}$@ADS41',
                    r'$E_{\mathrm{ads}}^{\mathrm{che}}$@ADS41']
    else:
        plt.ylabel(tag, fontsize=fz)
    plt.xticks([r + barWidth*3.9 for r in br1], datasets, rotation=45, fontsize=fz)
    plt.yticks(fontsize=fz)
    if tag == 'MAE':
        plt.ylim([0.0,0.6])
    if tag == 'ME':
        plt.ylim([-0.55,0.55])
    plt.legend(fontsize=fz,ncol=3)


    plt.subplots_adjust(left=0.08,
                        bottom=0.23,
                        right=0.99,
                        top=0.97,
                        wspace=0.0,
                        hspace=0.00)
    # Save figure, but also show it
    plt.savefig('pics/datasets_comp_{}.png'.format(tag),bbox_inches='tight',pad_inches = 0, dpi=300)
    plt.show()



def main(print_values=True,print_systems=True,print_valuesAndSystems=True,print_errors=True,plot_barplots=True,plot_allErrors=True):
    '''
        get all values for all functionals

        Do specific evaluations:
            print_values:           Prints all values of all functionals in a Latex-format
            print_systems:          Prints all systems of a set in a Latex-Format
            print_valuesAndSystems: Prints both
            print_errors:           Calculate errors and print them on screen
            plot_barplots:          Plot barplots for the ME, MAE, and MAPE
            plot_allErrors:         Plot errors for each dataset per functional
    '''
    pbe         = values()
    pbed3       = values()
    ms2         = values()
    scan        = values()
    r2scan      = values()
    scanrvv10   = values()
    mcml        = values()
    mcmlrvv10   = values()
    vcmlrvv10   = values()
    ref         = values()

    pbe.get_values('PBE')
    pbed3.get_values('PBE-D3')
    ms2.get_values('MS2')
    scan.get_values('SCAN')
    r2scan.get_values('r2SCAN')
    scanrvv10.get_values('SCAN-rVV10')
    mcml.get_values('MCML')
    mcmlrvv10.get_values('MCML-rVV10')
    vcmlrvv10.get_values('VCML-rVV10')
    ref.get_values('REF')

    # Order: PBE, PBE-D3, MS2, SCAN, R2SCAN, SCAN-rVV10, MCML, MCML-rVV10, VCML-rVV10
    if print_values == True:
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10],ref,tag='DBH24')
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10],ref,tag='RE42')
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10,ref],ref,tag='S66x8') # add ref, so to print a table with reference values
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10],ref,tag='W4-11')
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10],ref,tag='alat')
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10],ref,tag='Ecoh')
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10],ref,tag='bulk')
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10],ref,tag='ADS41')

    if print_systems == True:
        make_table_system(ref,tag='DBH24')
        make_table_system(ref,tag='RE42')
        make_table_system(ref,tag='S66x8')
        make_table_system(ref,tag='W4-11')
        make_table_system(ref,tag='SOL62')
        make_table_system(ref,tag='ADS41')

    if print_valuesAndSystems == True:
        make_table_system(ref,tag='DBH24')
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10],ref,tag='DBH24')
        make_table_system(ref,tag='RE42')
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10],ref,tag='RE42')
        make_table_system(ref,tag='S66x8')
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10,ref],ref,tag='S66x8')
        make_table_system(ref,tag='W4-11')
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10],ref,tag='W4-11')
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10],ref,tag='alat')
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10],ref,tag='Ecoh')
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10],ref,tag='bulk')
        make_table_system(ref,tag='ADS41')
        make_table_values([pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10],ref,tag='ADS41')

    if print_errors == True:
        all_calc = [pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10]
        calc_errors(all_calc,ref)

    if plot_barplots == True:
        all_calc = [pbe,pbed3,ms2,scan,r2scan,scanrvv10,mcml,mcmlrvv10,vcmlrvv10]
        plot_barPlots(all_calc,ref,tag='MAE')
        plot_barPlots(all_calc,ref,tag='ME')
        plot_barPlots(all_calc,ref,tag='MAPE')

    if plot_allErrors == True:
        # Plot in groups
        # PBE and PBE-D3 (only GGA)
        # MS2 and MCML (MCML is based on MS2)
        # SCAN and r2SCAN
        # SCAN-rVV10, MCML-rVV10 and VCML-rVV10
        all_calc = [pbe,pbed3]
        plot_errors(all_calc,ref)
        all_calc = [ms2,mcml]
        plot_errors(all_calc,ref)
        all_calc = [scan,r2scan]
        plot_errors(all_calc,ref)
        all_calc = [scanrvv10,mcmlrvv10,vcmlrvv10]
        plot_errors(all_calc,ref)


if __name__ == '__main__':
    '''
       print_values:           Prints all values of all functionals in a Latex-format
       print_systems:          Prints all systems of all set in a Latex-Format
       print_valuesAndSystems: Prints both
       print_errors:           Calculate errors and print them on screen
       plot_barplots:          Plot barplots for the ME, MAE, and MAPE
       plot_allErrors:         Plot errors for each dataset per functional, save figures in /pics
    '''
    print_values            = False
    print_systems           = False
    print_valuesAndSystems  = False # print both, systems and values, right after each other
    print_errors            = True
    plot_barplots           = True
    plot_allErrors          = False
    main(print_values=print_values,
         print_systems=print_systems,
         print_valuesAndSystems=print_valuesAndSystems,
         print_errors=print_errors,
         plot_barplots=plot_barplots,
         plot_allErrors=plot_allErrors)

