import subprocess
import os
import numpy as np
from ase.io import read,write
from ase.spacegroup import get_spacegroup,Spacegroup
from ase.cluster import Icosahedron, Octahedron, Decahedron
from diffpy.srfit.fitbase import FitContribution, FitRecipe
from diffpy.srfit.fitbase import FitResults
from diffpy.srfit.fitbase import Profile
from diffpy.srfit.pdf import PDFParser, DebyePDFGenerator
from diffpy.structure import Structure
from scipy.optimize import least_squares
import matplotlib as mpl
from matplotlib import pyplot as plt
from pathlib import Path
import glob
import math
import re
import random
from ase.calculators.emt import EMT
from ase.optimize import BFGS,GPMin,FIRE,MDMin
from ase.io.trajectory import Trajectory
from scipy.spatial import ConvexHull, cKDTree
from ase import Atoms


class PDFExtractor:
    def __init__(self,
                 datafilelist,
                 composition, 
                 qmin,
                 qmax,
                 qmaxinst,
                 wavelength=0.7107,
                 dataformat='QA',
                 rmin=0,
                 rmax=50,
                 rstep=0.01,
                 bgscale=1,
                 rpoly=0.9,
                 emptyfile=None):
        self.datafilelist=datafilelist
        self.emptyfile=emptyfile
        self.composition=composition
        self.qmin=qmin
        self.qmax=qmax
        self.qmaxinst=qmaxinst
        self.wl=wavelength
        self.dataformat=dataformat
        self.rmin=rmin
        self.rmax=rmax
        self.rstep=rstep
        self.bgscale=bgscale
        self.rpoly=rpoly


    def writecfg(self):
        """
        datafilelist: list of paths to data files from wich PDF should be extracted
        """

        self.datapath=os.path.dirname(self.datafilelist[0])
        self.pdfpath=self.datapath+'/extracted_PDF'
        
        os.makedirs(self.pdfpath,exist_ok=True)

        cfg=open(self.pdfpath+'/pdfgetX3_GUI.cfg','w')
        cfg.write('[DEFAULT] \n')
        cfg.write('dataformat = %s' %self.dataformat +' \n')
        
        
        
        cfg.write('inputfile='+''.join(os.path.basename(i) +'\n' +'\t'
                                for i in self.datafilelist[:-1]))
        cfg.write('\t %s' %os.path.basename(self.datafilelist[-1])+'\n')
        cfg.write('datapath = %s' % os.path.dirname(self.datafilelist[0])+'/' +'\n')
        if self.emptyfile is not None:
            cfg.write('\t %s' %os.path.dirname(self.emptyfile)+'\n')

            cfg.write('bgscale=%f \n' %self.bgscale)
            cfg.write('backgroundfile=%s' % os.path.basename(self.emptyfile)+'\n')
        
            
        cfg.write('composition= %s \n'%str(self.composition))
        cfg.write('qmin=%f \n' %self.qmin)
        cfg.write('qmax=%f \n' %self.qmax)
        cfg.write('qmaxinst=%f \n' %self.qmaxinst)
        cfg.write('wavelength=%f \n' %self.wl)
        cfg.write('mode = xray \n')
        cfg.write('rpoly=%f \n' %self.rpoly)
        cfg.write('rmin=%f \n' %self.rmin)
        cfg.write('rstep=%f \n' %self.rstep)
        cfg.write('rmax=%f \n' %self.rmax)       
        cfg.write('output=%s' %self.pdfpath +'/@b.@o \n')
        cfg.write('outputtype = sq,gr \n')
        #cfg.write('plot = iq,fq,gr \n' )
        cfg.write('force = yes \n')
        
        cfg.close()
        return
    

    def extractpdf(self):
        self.writecfg()
        command = 'conda run -n py36 pdfgetx3 -c' +self.pdfpath+'/pdfgetX3_GUI.cfg'

        # Use subprocess to execute the command
        subprocess.run(command, shell=True)
        print(f'PDF file(s) extracted in {self.pdfpath}')
        # Plot pdf
        
        fig,ax=plt.subplots()
        for file in self.datafilelist:
            rootname=(os.path.basename(file).split('/')[-1]).split('.')[0]
            pdffile=self.pdfpath+f'/{rootname}.gr'
            r,g=np.loadtxt(pdffile,skiprows=27,unpack=True)
            ax.plot(r,g,label=rootname)
        ax.set_xlabel('r ($\\AA$)')
        ax.set_ylabel('G(r)')
        fig.legend()
        fig.tight_layout()

        return self.pdfpath
    

class StructureGenerator():
    def __init__(self,pdfpath,cif_file:str,size_array:tuple=None, min_params:tuple=[1,1],max_params:tuple=[10,10],sphere_only: bool=False,
                 auto_mode: bool=False, pdf_file: str=None, r_coh: float=None, n_sizes: int=2, tolerance: float=0.1,
                 max_search_param: int=20, derivative_sigma: float=5.0, amplitude_sigma: float=3.0, 
                 window_size: int=10, derivative_weight: float=0.0, noise_window_start: float=0.85, 
                 score_threshold: float=0.001, n_jobs: int=-1):
        """
        pdfpath: directory where pdf are stored
        cif_file: path to cif file (provide Fm-3m SG if N.A. (e.g. icosahedra))
        size_array: tuple array of diameters of envelopping sphere (if None and auto_mode=True, will be auto-determined)
        min_params: tuple array of parameters used to define ase clusters (min values) - ignored in auto_mode
        max_params: tuple array of parameters used to define ase clusters (max values) - ignored in auto_mode
        sphere_only: bool Make Spherical particles only
        auto_mode: bool If True, automatically determine sizes from PDF analysis or r_coh
        pdf_file: str Path to PDF file to analyze (required if auto_mode=True and r_coh=None)
        r_coh: float Coherence length / max particle diameter (Å). If provided, bypasses automatic detection
        n_sizes: int Number of different sizes to generate for spheres in auto mode (default=2)
        tolerance: float Relative tolerance around r_coh (±tolerance*100%)
        max_search_param: int Maximum value for p and q parameters when searching in auto mode
        derivative_sigma: float Sigma multiplier for derivative threshold (for auto detection only)
        amplitude_sigma: float Sigma multiplier for amplitude threshold (for auto detection only)
        window_size: int Window size for local statistics in PDF analysis (for auto detection only)
        derivative_weight: float Weight for derivative terms (default=0.0, set >0 to use derivative; if 0, only amplitude is used)
        noise_window_start: float Fraction of r-range where noise reference window starts (default=0.85 = last 15%)
        score_threshold: float Score threshold for r_max detection (default=0.001). Lower values = stricter detection (larger r_max)
        n_jobs: int Number of parallel jobs for structure generation (-1 = all CPU cores, 1 = sequential)
        """
        self.pdfpath=pdfpath
        self.cif_file=cif_file
        self.auto_mode=auto_mode
        self.pdf_file=pdf_file
        self.r_coh=r_coh
        self.n_sizes=n_sizes
        self.tolerance=tolerance
        self.max_search_param=max_search_param
        self.derivative_sigma=derivative_sigma
        self.amplitude_sigma=amplitude_sigma
        self.window_size=window_size
        self.derivative_weight=derivative_weight
        self.noise_window_start=noise_window_start
        self.score_threshold=score_threshold
        self.n_jobs=n_jobs
        
        # Auto mode: determine parameters from r_coh or PDF analysis
        if self.auto_mode:
            if self.r_coh is not None:
                # Manual specification of coherence length
                print(f"Using user-specified r_coh: {self.r_coh:.2f} Å")
                self.r_max = self.r_coh
            else:
                # Automatic detection from PDF (fallback if no r_coh)
                if self.pdf_file is not None:
                    print("No r_coh specified, analyzing PDF to auto-detect r_max...")
                    try:
                        self.r_max = self.analyze_pdf_and_get_rmax()
                    except Exception as e:
                        print(f"⚠️  Auto-detection failed: {e}")
                        print("Using default r_max = 30.0 Å")
                        self.r_max = 30.0
                else:
                    # No r_coh and no pdf_file: use default
                    print("⚠️  No r_coh or pdf_file provided, using default r_max = 30.0 Å")
                    self.r_max = 30.0
            
            self.size_array = self.auto_size_array_from_rmax()
        else:
            if size_array is None:
                raise ValueError("size_array must be provided when auto_mode=False")
            self.size_array=size_array
            self.r_max = None
        
        self.structure=read(self.cif_file)
        #SG=Spacegroup(structure)
        SG=Spacegroup(get_spacegroup(self.structure))
        
        self.SGNo=SG.no
        self.lattice_parameters=self.structure.get_cell()
        self.a,self.b,self.c=self.lattice_parameters.lengths()
        self.alpha,self.beta,self.gamma=self.lattice_parameters.angles()
        self.atoms=self.structure.get_chemical_symbols()
        self.atom_positions=self.structure.get_scaled_positions()
        self.bravais=self.get_crystal_type()
        print('Crystal structure loaded from cif:')
        print(f'Cell edges: a={self.a:4f}, b={self.b:4f}, c={self.c:4f}')
        print(f'Cell angles: $\\alpha$={self.alpha:.2f},$\\beta$={self.beta:.2f}, $\\gamma$={self.gamma:.2f} ')
        print(f'Bravais unit cell:{self.bravais}')     
        print('Atomic Positions:')
        i=0
        for frac_coord in enumerate(self.atom_positions):
            print(f"Atom {self.atoms[i]}: {frac_coord}")
            i+=1
        pass
        self.min_params=min_params
        self.max_params=max_params
        self.sphere_only=sphere_only
    
    def analyze_pdf_and_get_rmax(self):
        """
        Analyzes PDF to find r_max where G(r) becomes noise
        Uses strong smoothing then derivative to detect signal->noise transition
        
        Returns:
            r_max: value of r where PDF becomes quasi-null
        """
        from scipy.signal import savgol_filter
        
        # Read PDF (assuming standard .gr format: r, G(r))
        try:
            data = np.loadtxt(self.pdf_file, skiprows=27)
            r = data[:, 0]
            gr = data[:, 1]
        except:
            raise ValueError(f"Cannot read PDF file: {self.pdf_file}")
        
        # STRONG SMOOTHING to eliminate FFT truncation oscillations
        # Use Savitzky-Golay with wide window
        window_length = min(101, len(gr) - 1)  # Must be odd
        if window_length % 2 == 0:
            window_length -= 1
        
        gr_smooth = savgol_filter(gr, window_length=window_length, polyorder=3)
        
        # Find position of main G(r) peak
        idx_max = np.argmax(np.abs(gr_smooth))
        r_max_peak = r[idx_max]
        
        # Calculate derivative of SMOOTHED signal
        dgr_smooth = np.gradient(gr_smooth, r)
        
        # Also smooth the derivative
        dgr_smooth = savgol_filter(dgr_smooth, window_length=window_length, polyorder=2)
        
        # Calculate threshold based on user-defined noise window
        noise_region_start = int(self.noise_window_start * len(r))
        dgr_noise = dgr_smooth[noise_region_start:]
        gr_noise = gr_smooth[noise_region_start:]
        
        # Noise statistics
        noise_deriv_std = np.std(dgr_noise)
        noise_gr_std = np.std(gr_noise)
        noise_gr_mean = np.mean(np.abs(gr_noise))
        
        # Reference values for scoring
        derivative_ref = noise_deriv_std * self.derivative_sigma
        gr_ref = noise_gr_mean + self.amplitude_sigma * noise_gr_std
        
        # Start search AFTER main peak
        # Use CONTINUOUS scoring instead of binary thresholds
        search_window = max(self.window_size, 20)
        
        # Define minimum search radius to avoid early local minima
        # Set to at least 15 Å to avoid detecting artifacts in the first coordination shells
        rmin_search = max(15.0, r_max_peak + 5.0)
        idx_min_search = np.argmin(np.abs(r - rmin_search))
        
        best_candidate = None
        all_scores = []
        
        # Start search from idx_min_search instead of idx_max
        for i in range(max(idx_max, idx_min_search), len(r) - search_window):
            window_deriv = dgr_smooth[i:i+search_window]
            window_gr = gr_smooth[i:i+search_window]
            
            # Window statistics
            std_deriv = np.std(window_deriv)
            mean_abs_gr = np.mean(np.abs(window_gr))
            max_abs_deriv = np.max(np.abs(window_deriv))
            max_abs_gr = np.max(np.abs(window_gr))
            
            # CONTINUOUS score: combination of normalized deviations from noise level
            # Lower score = closer to noise = better candidate
            if self.derivative_weight == 0:
                # Simplified: only use amplitude (most effective according to user tests)
                score = (mean_abs_gr / (gr_ref + 1e-10))**2
            else:
                # Full score with derivative terms weighted
                score = (
                    self.derivative_weight * (max_abs_deriv / (derivative_ref + 1e-10))**2 +
                    (mean_abs_gr / (gr_ref + 1e-10))**2 +
                    self.derivative_weight * ((std_deriv - noise_deriv_std) / (noise_deriv_std + 1e-10))**2
                )
            
            all_scores.append((i, r[i], score))
            
            # NEW STRATEGY: Find FIRST position where score < threshold (not absolute minimum)
            # This makes the detection sensitive to gr_ref value
            if best_candidate is None and score < self.score_threshold:
                best_candidate = i
                detected_score = score
                break  # Stop at first match
        
        # Fallback: if no position meets threshold, use minimum score
        if best_candidate is None:
            min_score = float('inf')
            for idx, r_val, score in all_scores:
                if score < min_score:
                    min_score = score
                    best_candidate = idx
            detected_score = min_score
        
        # Always return best candidate found
        if best_candidate is not None:
            print(f"  r_max detected: {r[best_candidate]:.2f} Å (score: {detected_score:.3f})")
            print(f"  Search started from r = {rmin_search:.1f} Å to avoid early artifacts")
            return r[best_candidate]
        
        # Fallback: return 70% of r_max if no clear detection
        print(f"  No clear transition detected, using 70% of r_max")
        return r[-1] * 0.7
    
    def plot_rmax_diagnostic(self, save_path=None):
        """
        Visual diagnostic for r_max detection
        Shows G(r), smoothed G(r), derivative, noise reference, and detected r_max
        
        Args:
            save_path: Optional path to save the figure
        """
        from scipy.signal import savgol_filter
        import matplotlib.pyplot as plt
        
        if self.pdf_file is None:
            print("No PDF file provided, cannot generate diagnostic")
            return
        
        # Read PDF
        try:
            data = np.loadtxt(self.pdf_file, skiprows=27)
            r = data[:, 0]
            gr = data[:, 1]
        except:
            print(f"Cannot read PDF file: {self.pdf_file}")
            return
        
        # Smoothing
        window_length = min(101, len(gr) - 1)
        if window_length % 2 == 0:
            window_length -= 1
        gr_smooth = savgol_filter(gr, window_length=window_length, polyorder=3)
        
        # Find main peak
        idx_max = np.argmax(np.abs(gr_smooth))
        r_max_peak = r[idx_max]
        
        # Calculate derivative
        dgr_smooth = np.gradient(gr_smooth, r)
        dgr_smooth = savgol_filter(dgr_smooth, window_length=window_length, polyorder=2)
        
        # Noise reference region
        noise_region_start = int(self.noise_window_start * len(r))
        dgr_noise = dgr_smooth[noise_region_start:]
        gr_noise = gr_smooth[noise_region_start:]
        
        noise_deriv_std = np.std(dgr_noise)
        noise_gr_std = np.std(gr_noise)
        noise_gr_mean = np.mean(np.abs(gr_noise))
        
        # Use same method as in analyze_pdf_and_get_rmax
        derivative_ref = noise_deriv_std * self.derivative_sigma
        gr_ref = noise_gr_mean + self.amplitude_sigma * noise_gr_std
        
        # Calculate score curve for visualization
        search_window = max(self.window_size, 20)
        score_curve = []
        r_search = []
        
        for i in range(idx_max, len(r) - search_window):
            window_deriv = dgr_smooth[i:i+search_window]
            window_gr = gr_smooth[i:i+search_window]
            
            std_deriv = np.std(window_deriv)
            mean_abs_gr = np.mean(np.abs(window_gr))
            max_abs_deriv = np.max(np.abs(window_deriv))
            
            if self.derivative_weight == 0:
                score = (mean_abs_gr / (gr_ref + 1e-10))**2
            else:
                score = (
                    self.derivative_weight * (max_abs_deriv / (derivative_ref + 1e-10))**2 +
                    (mean_abs_gr / (gr_ref + 1e-10))**2 +
                    self.derivative_weight * ((std_deriv - noise_deriv_std) / (noise_deriv_std + 1e-10))**2
                )
            score_curve.append(score)
            r_search.append(r[i])
        
        score_curve = np.array(score_curve)
        r_search = np.array(r_search)
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 15))
        fig.suptitle('r_max Detection Diagnostic', fontsize=14, fontweight='bold')
        
        # Subplot 1: G(r) raw vs smoothed
        ax1 = axes[0]
        ax1.plot(r, gr, 'gray', alpha=0.3, label='Raw G(r)', linewidth=0.5)
        ax1.plot(r, gr_smooth, 'b-', label='Smoothed G(r)', linewidth=2)
        ax1.axvline(r_max_peak, color='green', linestyle='--', alpha=0.7, label=f'Main peak (r={r_max_peak:.2f} Å)')
        if self.r_max is not None:
            ax1.axvline(self.r_max, color='red', linestyle='-', linewidth=2, label=f'r_max detected ({self.r_max:.2f} Å)')
        ax1.axvspan(r[noise_region_start], r[-1], alpha=0.2, color='orange', label=f'Noise reference (last {(1-self.noise_window_start)*100:.0f}%)')
        ax1.axhline(gr_ref, color='purple', linestyle=':', alpha=0.7, label=f'Amplitude ref ({gr_ref:.4f})')
        ax1.axhline(-gr_ref, color='purple', linestyle=':', alpha=0.7)
        ax1.set_xlabel('r (Å)', fontsize=11)
        ax1.set_ylabel('G(r)', fontsize=11)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('PDF Signal Analysis', fontsize=11)
        
        # Subplot 2: Derivative dG/dr
        ax2 = axes[1]
        ax2.plot(r, dgr_smooth, 'darkblue', linewidth=1.5, label='dG/dr (smoothed)')
        ax2.axvline(r_max_peak, color='green', linestyle='--', alpha=0.7, label=f'Main peak')
        if self.r_max is not None:
            ax2.axvline(self.r_max, color='red', linestyle='-', linewidth=2, label=f'r_max detected ({self.r_max:.2f} Å)')
        ax2.axvspan(r[noise_region_start], r[-1], alpha=0.2, color='orange', label='Noise reference')
        ax2.axhline(derivative_ref, color='purple', linestyle=':', alpha=0.7, label=f'Derivative ref ({derivative_ref:.4f})')
        ax2.axhline(-derivative_ref, color='purple', linestyle=':', alpha=0.7)
        ax2.axhline(0, color='k', linestyle='-', alpha=0.3, linewidth=0.5)
        ax2.set_xlabel('r (Å)', fontsize=11)
        ax2.set_ylabel('dG/dr', fontsize=11)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Derivative Analysis', fontsize=11)
        
        # Subplot 3: Absolute values with detection window
        ax3 = axes[2]
        ax3.plot(r, np.abs(gr_smooth/gr_smooth.max()), 'b-', linewidth=1.5, label='|G(r)| smoothed')
        ax3.plot(r, np.abs(dgr_smooth/dgr_smooth.max()), 'darkblue', linewidth=1.5, alpha=0.7, label='|dG/dr| smoothed')
        ax3.axvline(r_max_peak, color='green', linestyle='--', alpha=0.7, label=f'Main peak')
        if self.r_max is not None:
            ax3.axvline(self.r_max, color='red', linestyle='-', linewidth=2, label=f'r_max detected ({self.r_max:.2f} Å)')
            # Show detection window
            search_window = max(self.window_size, 20)
            idx_rmax = np.argmin(np.abs(r - self.r_max))
            if idx_rmax + search_window < len(r):
                ax3.axvspan(self.r_max, r[idx_rmax + search_window], alpha=0.15, color='red', label=f'Detection window ({search_window} pts)')
        ax3.axvspan(r[noise_region_start], r[-1], alpha=0.2, color='orange', label='Noise reference')
        # Normalized references for log plot
        gr_ref_norm = gr_ref / np.max(np.abs(gr_smooth))
        deriv_ref_norm = derivative_ref / np.max(np.abs(dgr_smooth))
        ax3.axhline(gr_ref_norm, color='purple', linestyle=':', alpha=0.7, label=f'G ref')
        ax3.axhline(deriv_ref_norm, color='purple', linestyle='--', alpha=0.7, label=f'dG/dr ref')
        ax3.set_xlabel('r (Å)', fontsize=11)
        ax3.set_ylabel('Absolute values', fontsize=11)
        ax3.set_yscale('log')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3, which='both')
        ax3.set_title('Absolute Values (log scale)', fontsize=11)
        
        # Subplot 4: Detection score curve
        ax4 = axes[3]
        ax4.plot(r_search, score_curve, 'darkgreen', linewidth=2, label='Detection score')
        if self.r_max is not None:
            idx_detected = np.argmin(np.abs(r_search - self.r_max))
            ax4.plot(r_search[idx_detected], score_curve[idx_detected], 'r*', markersize=15, 
                    label=f'Detected r_max ({self.r_max:.2f} Å, score={score_curve[idx_detected]:.3f})')
        ax4.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Score = 1.0 (reference)')
        ax4.set_xlabel('r (Å)', fontsize=11)
        ax4.set_ylabel('Detection Score', fontsize=11)
        ax4.set_yscale('log')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3, which='both')
        ax4.set_title('Continuous Score (minimum = detected r_max)', fontsize=11)
        ax4.set_ylim(bottom=0.01)
        
        plt.tight_layout()
        
        # Print diagnostic info
        print("\n" + "="*60)
        print("R_MAX DETECTION DIAGNOSTIC")
        print("="*60)
        print(f"Smoothing window: {window_length} points")
        print(f"Main peak position: {r_max_peak:.2f} Å")
        print(f"Search starts after: {r_max_peak:.2f} Å")
        print(f"\nNoise reference statistics (from {self.noise_window_start*100:.0f}% to 100% of r-range):")
        print(f"  - std(dG/dr): {noise_deriv_std:.4f}")
        print(f"  - std(G): {noise_gr_std:.4f}")
        print(f"  - mean(|G|): {noise_gr_mean:.4f}")
        print(f"\nReference values (for scoring):")
        print(f"  - Derivative ref: {derivative_ref:.4f} ({self.derivative_sigma}σ)")
        print(f"  - Amplitude ref: {gr_ref:.4f} (mean + {self.amplitude_sigma}σ)")
        print(f"\nDetection method: Continuous score minimization")
        print(f"  Lower score = closer to noise level = better candidate")
        print(f"\nDetection parameters:")
        print(f"  - derivative_sigma: {self.derivative_sigma}")
        print(f"  - amplitude_sigma: {self.amplitude_sigma}")
        print(f"  - window_size: {self.window_size}")
        print(f"  - derivative_weight: {self.derivative_weight} {'(amplitude only)' if self.derivative_weight == 0 else '(derivative + amplitude)'}")
        print(f"  - noise_window_start: {self.noise_window_start} (noise reference from {self.noise_window_start*100:.0f}% of r-range)")
        if self.r_max is not None:
            print(f"\nDetected r_max: {self.r_max:.2f} Å")
            print(f"Target diameter range: [{self.r_max*(1-self.tolerance):.2f}, {self.r_max*(1+self.tolerance):.2f}] Å")
        print("="*60)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nDiagnostic plot saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def auto_size_array_from_rmax(self):
        """
        Génère automatiquement size_array basé sur r_max et la tolérance
        
        Returns:
            size_array: tuple de diamètres pour les sphères
        """
        r_min = self.r_max * (1 - self.tolerance)
        r_max_tol = self.r_max * (1 + self.tolerance)
        
        size_array = tuple(np.linspace(r_min, r_max_tol, self.n_sizes))
        
        return size_array
    
    def is_diameter_in_target_range(self, diameter):
        """
        Vérifie si un diamètre tombe dans la fenêtre cible
        
        Args:
            diameter: diamètre à vérifier
            
        Returns:
            bool: True si le diamètre est dans la fenêtre acceptable
        """
        if self.r_max is None:
            return True  # Pas de filtre en mode manuel
        
        d_min = self.r_max * (1 - self.tolerance)
        d_max = self.r_max * (1 + self.tolerance)
        
        return d_min <= diameter <= d_max
        
    def get_crystal_type(self):
        """
        Find the Bravais lattice based on the space group number.
        """  
        spacegroup_number=self.SGNo
        # bravais lattice based on space group number https://fr.wikipedia.org/wiki/Groupe_d%27espace
        if 195 <= spacegroup_number <= 230:  # Cubic
            if spacegroup_number == 225:
                return 'fcc'
            elif spacegroup_number == 229:
                return 'bcc'
            else:
                return 'cubic'
        elif 168 <= spacegroup_number <= 194:  # Hexagonal
            return 'hcp'
        elif 75 <= spacegroup_number <= 142:  # Tetragonal
            return 'tetragonal'
        elif 16 <= spacegroup_number <= 74:  # Orthorhombic
            return 'orthorhombic'
        elif 3 <= spacegroup_number <= 15:  # Monoclinic
            return 'monoclinic'
        elif 1 <= spacegroup_number <= 2:  # Triclinic
            return 'triclinic'
        else:
            return 'unknown'

    def diameter_from_Atoms(self,Atoms):
        xyz_coord=Atoms.get_positions()
        x=list(zip(*xyz_coord))[0];y=list(zip(*xyz_coord))[1];z=list(zip(*xyz_coord))[2]
        x_center=np.mean(x);y_center=np.mean(y);z_center=np.mean(z)
        x_ok=x-x_center;y_ok=y-y_center;z_ok=z-z_center
        r=(x_ok**2+y_ok**2+z_ok**2)**(1/2)
        return max(r)  

    def center(self,pos_array):
        output=np.zeros_like(pos_array)
        x=pos_array[:,0];y=pos_array[:,1];z=pos_array[:,2]
        x0=np.mean(x);y0=np.mean(y);z0=np.mean(z)
        i=0
        for pos in pos_array:
            x,y,z=pos
            xok=x-x0;yok=y-y0;zok=z-z0
            output[i]=[xok,yok,zok]
            i+=1
        return output

    def writexyz(self,filename,atoms):
        """atoms ase Atoms object"""
        cifname=(os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
        strufile_dir=self.pdfpath+f'/structure_files_{cifname}'
        os.makedirs(strufile_dir,exist_ok=True)
        #write(strufile_dir+f'/{filename}.xyz',atoms)
        element_array=atoms.get_chemical_symbols()
        # extract composition in dict form
        composition={}
        for element in element_array:
            if element in composition:
                composition[element]+=1
            else:
                composition[element]=1
        
        coord=atoms.get_positions()
        natoms=len(element_array)  
        line2write='%d \n'%natoms
        line2write+='%s\n'%str(composition)
        for i in range(natoms):
            line2write+='%s'%str(element_array[i])+'\t %.8f'%float(coord[i,0])+'\t %.8f'%float(coord[i,1])+'\t %.8f'%float(coord[i,2])+'\n'
        with open(strufile_dir+f'/{filename}.xyz','w') as file:
            file.write(line2write)

    def makeSphere(self,phi):
        # makesupercell
        nbcell=np.max([math.ceil(phi/self.a),math.ceil(phi/self.b),math.ceil(phi/self.c)])+1
        scaling_factors=[nbcell,nbcell,nbcell]
        supercell = self.structure.repeat(scaling_factors)
        
        original_positions = supercell.get_positions()

        #positions should be centered around 0
        original_positions=self.center(original_positions)
        atom_names=supercell.get_atomic_numbers()
        
        # atoms to delete
        delAtoms=[]
        for i in range(len(atom_names)):            
            x, y, z = original_positions[i]            
            r = np.sqrt(x**2 + y**2+z**2)
            condition=True
            # Ensure the cylinder is maintained
            if r > phi/2:
                condition=False
            if not condition:
                delAtoms.append(i)
        del supercell[delAtoms]
        nbatoms=len(supercell)
        #write xyz file
        cifname=(os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
        filename=f'Sphere_phi={int(phi)}_{cifname}_{nbatoms}atoms'
        self.writexyz(filename,supercell)
        return filename,phi,nbatoms
    
    def makeIcosahedron(self,p):
        ico=Icosahedron(self.atoms[0],p,self.a)
        nbatoms=len(ico)
        cifname=(os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
        filename=f'Ih_{p}shells_phi={int(2*self.diameter_from_Atoms(ico))}_{cifname}_{nbatoms}atoms'
        self.writexyz(filename,ico)
        return filename,2*self.diameter_from_Atoms(ico),nbatoms
    
    def makeDecahedron(self,p,q):
        deca=Decahedron(self.atoms[0],p,q,0,self.a)
        nbatoms=len(deca)
        cifname=(os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
        filename=f'Dh_{p}_{q}_phi={int(2*self.diameter_from_Atoms(deca))}_{cifname}_{nbatoms}atoms'
        self.writexyz(filename,deca)
        return filename,2*self.diameter_from_Atoms(deca),nbatoms
    
    def makeOctahedron(self,p,q):
        
        octa=Octahedron(self.atoms[0],p,q,self.a)
        nbatoms=len(octa)
        cifname=(os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
        if q==0:
            filename=f'RegOh_{p}_0_phi={int(2*self.diameter_from_Atoms(octa))}_{cifname}_{nbatoms}atoms'
        if p==2*q+1:
            filename=f'CubOh_{p}_{q}_phi={int(2*self.diameter_from_Atoms(octa))}_{cifname}_{nbatoms}atoms'
        if p==3*q+1:
            filename=f'RegTrOh_{p}_{q}_phi={int(2*self.diameter_from_Atoms(octa))}_{cifname}_{nbatoms}atoms'
        else:
            filename=f'TrOh_{p}_{q}_phi={int(2*self.diameter_from_Atoms(octa))}_{cifname}_{nbatoms}atoms'
        self.writexyz(filename,octa)
        return filename,2*self.diameter_from_Atoms(octa),nbatoms
        
    def returnPointsThatLieInPlanes(self,planes: np.ndarray,
                                coords: np.ndarray,
                                debug: bool=False,
                                threshold: float=1e-3
                                ):
        """
        Finds all points (atoms) that lie within the given planes based on a signed distance criterion.

        Args:
            planes (np.ndarray): A 2D array where each row represents a plane equation [a, b, c, d] for the plane ax + by + cz + d = 0.
            coords (np.ndarray): A 2D array where each row is the coordinates of an atom [x, y, z].
            debug (bool, optional): If True, prints additional debugging information. Defaults to False.
            threshold (float, optional): The tolerance for the distance to the plane to consider a point as lying in the plane. Defaults to 1e-3.
            noOutput (bool, optional): If True, suppresses the output messages. Defaults to False.

        Returns:
            np.ndarray: A boolean array where True indicates that the atom lies in one of the planes.
        """
        import numpy as np
        
        AtomsInPlane = np.zeros(len(coords), dtype=bool)
        for p in planes:
            for i,c in enumerate(coords):
                signedDistance = self.Pt2planeSignedDistance(p,c)
                AtomsInPlane[i] = AtomsInPlane[i] or np.abs(signedDistance) < threshold
            nOfAtomsInPlane = np.count_nonzero(AtomsInPlane)
            if debug:
                print(f"- plane", [f"{x: .2f}" for x in p],f"> {nOfAtomsInPlane} atoms lie in the planes")
                for i,a in enumerate(delAtoms):
                    if a: print(f"@{i+1}",end=',')
                print("",end='\n')
        AtomsInPlane = np.array(AtomsInPlane)
        return AtomsInPlane

    def Pt2planeSignedDistance(self,plane,point):
        '''
        Returns the orthogonal distance of a given point X0 to the plane p in a metric space (projection of X0 on p = P), 
        with the sign determined by whether or not X0 is in the interior of p with respect to the center of gravity [0 0 0]
        Args:
            - plane (numpy array): [u v w h] definition of the P plane 
            - point (numpy array): [x0 y0 z0] coordinates of the X0 point 
        Returns:
            the signed modulus ±||PX0||
        '''
    
        sd = (plane[3] + np.dot(plane[0:3],point))/np.sqrt(plane[0]**2+plane[1]**2+plane[2]**2)
        return sd

    def coreSurface(self,atoms: Atoms,
                threshold=1e-3               
               ):       
    
        from scipy.spatial import ConvexHull
        
        coords = atoms.get_positions()
        hull = ConvexHull(coords)
        atoms.trPlanes = hull.equations
        surfaceAtoms = self.returnPointsThatLieInPlanes(atoms.trPlanes,coords,threshold=threshold)
    
        return [hull.vertices,hull.simplices,hull.neighbors,hull.equations], surfaceAtoms
    
    
    def detect_surface_atoms(self,filename,view=False):
        atoms=read(filename+'.xyz')
        _, surfaceAtoms = self.coreSurface(atoms)
        coords = atoms.get_positions()
        hull = ConvexHull(coords)
        surface_indices = hull.vertices
        n_surface_atoms = len(hull.vertices)
        if view:
            from ase.visualize import view
            surface_indices = hull.vertices

            # Create a copy to modify
            atoms_copy = atoms.copy()

            # Option 1: Change color by changing chemical symbols
            # For example, make surface atoms 'O' and others 'C'
            # (you can pick other symbols if you like)
            symbols = ['C'] * len(atoms)
            for idx in surface_indices:
                symbols[idx] = 'O'  # change to oxygen, so it'll show up red
            atoms_copy.set_chemical_symbols(symbols)

            view(atoms_copy)
        return surfaceAtoms.sum()

    def _process_icosahedron(self, p):
        """
        Traite un icosaèdre avec paramètre p (pour parallélisation)
        Retourne None si hors fenêtre, sinon (diameter, filename, size, nbatoms, nbsurfatoms)
        """
        try:
            ico = Icosahedron(self.atoms[0], p, self.a)
            diameter = 2 * self.diameter_from_Atoms(ico)
            
            if self.is_diameter_in_target_range(diameter):
                cifname = (os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
                strufile_dir = self.pdfpath + f'/structure_files_{cifname}/'
                filename, size, nbatoms = self.makeIcosahedron(p)
                nbsurfatoms = self.detect_surface_atoms(strufile_dir + filename)
                return (diameter, filename, size, nbatoms, nbsurfatoms)
        except Exception as e:
            pass
        return None
    
    def _process_decahedron(self, p, q):
        """
        Traite un décaèdre avec paramètres p, q (pour parallélisation)
        Retourne None si hors fenêtre, sinon (diameter, filename, size, nbatoms, nbsurfatoms)
        """
        try:
            deca = Decahedron(self.atoms[0], p, q, 0, self.a)
            diameter = 2 * self.diameter_from_Atoms(deca)
            
            if self.is_diameter_in_target_range(diameter):
                cifname = (os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
                strufile_dir = self.pdfpath + f'/structure_files_{cifname}/'
                filename, size, nbatoms = self.makeDecahedron(p, q)
                nbsurfatoms = self.detect_surface_atoms(strufile_dir + filename)
                return (diameter, filename, size, nbatoms, nbsurfatoms)
        except Exception as e:
            pass
        return None
    
    def _process_octahedron(self, p, q):
        """
        Traite un octaèdre avec paramètres p, q (pour parallélisation)
        Retourne None si hors fenêtre, sinon (diameter, filename, size, nbatoms, nbsurfatoms)
        """
        try:
            octa = Octahedron(self.atoms[0], p, q, self.a)
            diameter = 2 * self.diameter_from_Atoms(octa)
            
            if self.is_diameter_in_target_range(diameter):
                cifname = (os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
                strufile_dir = self.pdfpath + f'/structure_files_{cifname}/'
                filename, size, nbatoms = self.makeOctahedron(p, q)
                nbsurfatoms = self.detect_surface_atoms(strufile_dir + filename)
                return (diameter, filename, size, nbatoms, nbsurfatoms)
        except Exception as e:
            pass
        return None
    

    def run(self):
        """
        Méthode de génération classique (mode manuel)
        """
        if self.auto_mode:
            return self.run_auto()
        
        cifname=(os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
        strufile_dir=self.pdfpath+f'/structure_files_{cifname}/'
        logfile=strufile_dir+'/structure_generation.log'
        line2write= '*****************************************************\n\n'
        line2write+='                STRUCTURE GENERATION                 \n\n'
        line2write+='*****************************************************\n\n'
        line2write+='Structure File                                   \tDiameter \tNumber of atoms \tNumber of surface atoms\n'
        print(line2write)
        if not self.sphere_only:
            p_array=np.arange(self.min_params[0],self.max_params[0])
            q_array=np.arange(self.min_params[1],self.max_params[1])
            for p in p_array:
                filename,size,nbatoms=self.makeIcosahedron(p)
                nbsurfatoms=self.detect_surface_atoms(strufile_dir+filename)
                
                print(f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}')
                line2write+=f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}\n'
                for q in q_array:
                    if q>=1:
                        filename,size,nbatoms=self.makeDecahedron(p,q)
                        nbsurfatoms=self.detect_surface_atoms(strufile_dir+filename)
                        
                        print(f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}')
                        line2write+=f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}\n'
                    if q<=(p-1)/2:
                        filename,size,nbatoms=self.makeOctahedron(p,q)
                        nbsurfatoms=self.detect_surface_atoms(strufile_dir+filename)
                        
                        print(f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}')
                        line2write+=f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}\n'
            for size in self.size_array:
                filename,size,nbatoms=self.makeSphere(size)
                nbsurfatoms=self.detect_surface_atoms(strufile_dir+filename)
                
                print(f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}')
                line2write+=f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}\n'
        else:
            for size in self.size_array:
                filename,size,nbatoms=self.makeSphere(size)
                nbsurfatoms=self.detect_surface_atoms(strufile_dir+filename)
                
                print(f'{filename:30}\t\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}')
                line2write+=f'{filename:30}\t\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}\n'
        with open(logfile,'w')as f:
            f.write(line2write)
        return strufile_dir
    
    def run_auto(self):
        """
        Automatic generation method based on PDF analysis
        Generates candidate structures and keeps only those within diameter window
        Uses multiprocessing for acceleration
        """
        from multiprocessing import Pool, cpu_count
        from tqdm import tqdm
        import os as os_module
        
        cifname=(os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
        strufile_dir=self.pdfpath+f'/structure_files_{cifname}/'
        logfile=strufile_dir+'/structure_generation.log'
        
        # Determine number of processes
        if self.n_jobs == -1:
            n_processes = cpu_count()
        else:
            n_processes = max(1, self.n_jobs)
        
        print(f"\nStarting structure generation with {n_processes} parallel processes")
        print(f"Target diameter range: [{self.r_max*(1-self.tolerance):.2f}, {self.r_max*(1+self.tolerance):.2f}] Å")
        
        line2write= '*****************************************************\n\n'
        line2write+='         STRUCTURE GENERATION (AUTO MODE)            \n\n'
        line2write+='*****************************************************\n\n'
        line2write+=f'PDF analyzed: {os.path.basename(self.pdf_file)}\n'
        line2write+=f'r_max detected: {self.r_max:.2f} Å\n'
        line2write+=f'Target diameter window: [{self.r_max*(1-self.tolerance):.2f}, {self.r_max*(1+self.tolerance):.2f}] Å\n'
        line2write+=f'Tolerance: ±{self.tolerance*100:.0f}%\n'
        line2write+=f'Parallel processes: {n_processes}\n\n'
        line2write+='Structure File                                   \tDiameter \tNumber of atoms \tNumber of surface atoms\n'
        
        structures_generated = []
        structures_kept = []
        kept_filenames = []  # Track filenames of kept structures
        results_to_write = []
        
        if not self.sphere_only:
            # Prepare tasks for icosahedra
            ico_tasks = [(p,) for p in range(1, self.max_search_param + 1)]
            
            # Process with progress bar
            print("\nSearching icosahedra...")
            if n_processes > 1:
                with Pool(processes=n_processes) as pool:
                    ico_results = list(tqdm(pool.starmap(self._process_icosahedron, ico_tasks), 
                                           total=len(ico_tasks), desc="Icosahedra", ncols=80))
            else:
                ico_results = [self._process_icosahedron(p) for p in tqdm(range(1, self.max_search_param + 1),
                                                                          desc="Icosahedra", ncols=80)]
            
            # Filter and store results
            for result in ico_results:
                if result is not None:
                    diameter, filename, size, nbatoms, nbsurfatoms = result
                    structures_generated.append(('Icosahedron', None, None, diameter))
                    if self.is_diameter_in_target_range(diameter):
                        structures_kept.append(('Icosahedron', None, None, diameter))
                        kept_filenames.append(strufile_dir + filename + '.xyz')
                        results_to_write.append(f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}\n')
            
            # Prepare tasks for decahedra
            deca_tasks = [(p, q) for p in range(1, self.max_search_param + 1) 
                         for q in range(1, self.max_search_param + 1)]
            
            print("Searching decahedra...")
            if n_processes > 1:
                with Pool(processes=n_processes) as pool:
                    deca_results = list(tqdm(pool.starmap(self._process_decahedron, deca_tasks),
                                            total=len(deca_tasks), desc="Decahedra", ncols=80))
            else:
                deca_results = [self._process_decahedron(p, q) for p, q in tqdm(deca_tasks,
                                                                                desc="Decahedra", ncols=80)]
            
            # Filter and store results
            for result in deca_results:
                if result is not None:
                    diameter, filename, size, nbatoms, nbsurfatoms = result
                    structures_generated.append(('Decahedron', None, None, diameter))
                    if self.is_diameter_in_target_range(diameter):
                        structures_kept.append(('Decahedron', None, None, diameter))
                        kept_filenames.append(strufile_dir + filename + '.xyz')
                        results_to_write.append(f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}\n')
            
            # Prepare tasks for octahedra
            octa_tasks = [(p, q) for p in range(1, self.max_search_param + 1) 
                         for q in range(0, (p-1)//2 + 1)]
            
            print("Searching octahedra...")
            if n_processes > 1:
                with Pool(processes=n_processes) as pool:
                    octa_results = list(tqdm(pool.starmap(self._process_octahedron, octa_tasks),
                                            total=len(octa_tasks), desc="Octahedra", ncols=80))
            else:
                octa_results = [self._process_octahedron(p, q) for p, q in tqdm(octa_tasks,
                                                                                desc="Octahedra", ncols=80)]
            
            # Filter and store results
            for result in octa_results:
                if result is not None:
                    diameter, filename, size, nbatoms, nbsurfatoms = result
                    structures_generated.append(('Octahedron', None, None, diameter))
                    if self.is_diameter_in_target_range(diameter):
                        structures_kept.append(('Octahedron', None, None, diameter))
                        kept_filenames.append(strufile_dir + filename + '.xyz')
                        results_to_write.append(f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}\n')
        
        # Generate spheres with auto-determined sizes
        print("Generating spheres...")
        for size in tqdm(self.size_array, desc="Spheres", ncols=80):
            filename,size,nbatoms=self.makeSphere(size)
            kept_filenames.append(strufile_dir + filename + '.xyz')  # Spheres always kept
            nbsurfatoms=self.detect_surface_atoms(strufile_dir+filename)
            results_to_write.append(f'{filename:50}\t{size:.4f}\t\t{nbatoms}\t\t\t{nbsurfatoms}\n')
        
        # Write all results to log
        line2write += ''.join(results_to_write)
        
        # Statistics
        line2write+='\n*****************************************************\n'
        line2write+=f'STATISTICS:\n'
        line2write+=f'Candidate structures tested: {len(structures_generated)}\n'
        line2write+=f'Structures kept: {len(structures_kept)} polyhedra + {len(self.size_array)} spheres\n'
        line2write+=f'Selection rate: {len(structures_kept)/max(len(structures_generated),1)*100:.1f}%\n'
        line2write+='*****************************************************\n'
        
        print('\n' + '='*60)
        print(f'GENERATION SUMMARY')
        print('='*60)
        print(f'Candidate structures tested: {len(structures_generated)}')
        print(f'Structures kept: {len(structures_kept)} polyhedra + {len(self.size_array)} spheres')
        print(f'Selection rate: {len(structures_kept)/max(len(structures_generated),1)*100:.1f}%')
        print(f'\nLog file: {logfile}')
        print('='*60)
        
        with open(logfile,'w')as f:
            f.write(line2write)
        
        # Save list of kept structures for screening
        kept_structures_file = strufile_dir + 'kept_structures.txt'
        with open(kept_structures_file, 'w') as f:
            for filename in kept_filenames:
                f.write(filename + '\n')
        
        return strufile_dir
        

class StructureCustom():
    def __init__ (self, 
                  strufile: str,
                  zoomscale:float = 1,
                  new_element: str =None,
                  fraction :float=0):
        """
        strufile: str, full path to structure file (xyz file)
        zoomscale: float, coefficient to adjust interatomic distance
        new_element: str, element to insert in the structure (randomly)
        fraction: float, fraction of the new element (between 0 and 1)
        """
        self.strufile=strufile
        self.path=os.path.dirname(self.strufile)
        self.zoomscale=zoomscale
        self.new_element=new_element
        self.fraction=fraction

    def apply_zoomscale(self):        
        self.x=[x*self.zoomscale for x in self.x]
        self.y=[y*self.zoomscale for y in self.y]
        self.z=[z*self.zoomscale for z in self.z]
        return self.x, self.y, self.z
    
    def parseline(self,line):
        parse=line.split('\t')
        element=parse[0];x=parse[1];y=parse[2];z=parse[3]
        return element,x,y,z
    
    def transform_structure(self):
        # extract data (element,x,y,z) from xyz file
        data=np.loadtxt(self.strufile,skiprows=2,dtype=[('element', 'U2'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        self.element=data['element']
        self.x=data['x'];self.y=data['y'];self.z=data['z']
        # apply zoomscale coefficient
        self.x, self.y, self.z=self.apply_zoomscale()
        
        # perform random substitution
        initial_elements=np.unique(self.element)
        initcompo=''
        for el in initial_elements:
            initcompo+=el
        N=len(self.element)
        k=N #number of initial elements
        if self.new_element is not None:           
            n=0 #number of new elements inserted in structure        
            while n<=(N*self.fraction):
                random_number = random.randint(0, N-1)
                if self.element[random_number] != self.new_element:
                    self.element[random_number]=self.new_element
                    n+=1
                    k-=1                    
                else: 
                    pass
                final_content='{%s'%initcompo+':%d'%k+',%s'%self.new_element+':%d}'%n
                outputfile=self.strufile.split('.')[0]+f'_zoomscale={self.zoomscale:.2f}_{initcompo}{100*(1-self.fraction):.0f}{self.new_element}{self.fraction*100:.0f}.xyz'
        else: # no random substitution
            final_content='{%s'%initcompo+':%d'%k+'}'
            outputfile=self.strufile.split('.')[0]+f'_zoomscale={self.zoomscale:.2f}.xyz'
        # write transformed structure to xyz file
        line2write=f'{N}\n{final_content}\n'
        for i in range(N):
            line2write += f"{self.element[i]} \t {self.x[i]:.4f} \t {self.y[i]:.4f} \t {self.z[i]:.4f} \n"
        
        with open(outputfile,'w') as f:
            f.write(line2write)
        return outputfile

    def optimize(self):
        xyzfile=self.strufile
        ico=read(xyzfile)
        ico.calc = EMT()
        basename=os.path.basename(xyzfile).split('/')[-1].split('.')[0]
        opt = FIRE(ico, trajectory=self.path+'/'+basename+'_FIRE.traj')
        opt.run(fmax=0.01)
        
        traj=Trajectory(self.path+'/'+basename+'_FIRE.traj')
        ico_opt=traj[-1]
        strufile_dir=self.path+f'/relaxed_structure_files/'
        os.makedirs(strufile_dir,exist_ok=True)
        outfilename=strufile_dir+basename+'_optimized.xyz'
        self.writexyz(outfilename,ico_opt)
        return outfilename
    
    def writexyz(self,filename,atoms):
        """atoms ase Atoms object"""
        #cifname=(os.path.basename(self.cif_file).split('/')[-1]).split('.')[0]
        
        
        #write(strufile_dir+f'/{filename}.xyz',atoms)
        element_array=atoms.get_chemical_symbols()
        # extract composition in dict form
        composition={}
        for element in element_array:
            if element in composition:
                composition[element]+=1
            else:
                composition[element]=1
        
        coord=atoms.get_positions()
        natoms=len(element_array)  
        line2write='%d \n'%natoms
        line2write+='%s\n'%str(composition)
        for i in range(natoms):
            line2write+='%s'%str(element_array[i])+'\t %.8f'%float(coord[i,0])+'\t %.8f'%float(coord[i,1])+'\t %.8f'%float(coord[i,2])+'\n'
        with open(f'/{filename}','w') as file:
            file.write(line2write)
    
    def view_structure(self, style='sphere', width=400, height=400, spin=True):
        """
        Visualize structure in 3D using py3Dmol in Jupyter notebook
        
        Parameters:
        -----------
        style: str, default='sphere'
            Visualization style: 'sphere', 'stick', 'cartoon', 'line', 'cross'
        width: int, default=400
            Width of the viewer in pixels
        height: int, default=400
            Height of the viewer in pixels
        spin: bool, default=True
            Enable automatic rotation
            
        Returns:
        --------
        view: py3Dmol.view object
        """
        try:
            import py3Dmol
        except ImportError:
            print("py3Dmol not installed. Install with: pip install py3Dmol")
            return None
        
        # Read structure data
        data = np.loadtxt(self.strufile, skiprows=2, 
                         dtype=[('element', 'U2'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        
        elements = data['element']
        coords = np.column_stack([data['x'], data['y'], data['z']]) * self.zoomscale
        
        # Create XYZ format string
        xyz_string = f"{len(elements)}\n"
        xyz_string += f"Structure with zoomscale={self.zoomscale}\n"
        for i, elem in enumerate(elements):
            xyz_string += f"{elem} {coords[i,0]:.6f} {coords[i,1]:.6f} {coords[i,2]:.6f}\n"
        
        # Create 3D viewer
        view = py3Dmol.view(width=width, height=height)
        view.addModel(xyz_string, 'xyz')
        
        # Apply style
        view.setStyle({style: {}})
        
        # Enable spin if requested
        if spin:
            view.spin(True)
        
        view.zoomTo()
        return view
    
    def save_structure_image(self, output_path, style='sphere', width=800, height=800):
        """
        Save structure visualization as PNG image
        
        Parameters:
        -----------
        output_path: str
            Path where to save the image (should end with .png)
        style: str, default='sphere'
            Visualization style
        width: int, default=800
            Image width in pixels
        height: int, default=800
            Image height in pixels
        """
        view = self.view_structure(style=style, width=width, height=height, spin=False)
        if view is not None:
            # Note: py3Dmol PNG export requires selenium/chromium
            # Alternative: use screenshot in notebook or export to other formats
            print(f"To save image, use: view.png() in notebook then save manually")
            print(f"Or use selenium for automated export")
            return view
        return None
    
    def get_structure_info(self):
        """
        Extract structure information including atom count and composition
        
        Returns:
        --------
        dict with keys: 'natoms', 'composition', 'elements'
        """
        data = np.loadtxt(self.strufile, skiprows=2, 
                         dtype=[('element', 'U2'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        
        elements = data['element']
        composition = {}
        for elem in elements:
            composition[elem] = composition.get(elem, 0) + 1
        
        return {
            'natoms': len(elements),
            'composition': composition,
            'elements': list(composition.keys()),
            'zoomscale': self.zoomscale
        }


class StructureReportGenerator():
    """
    Generate comprehensive HTML/PDF reports for structure screening results
    """
    
    def __init__(self, strufile_dir, best_results, screening_log=None, all_screening_results=None):
        """
        Parameters:
        -----------
        strufile_dir: str
            Directory containing structure files and generation log
        best_results: dict
            Results from StructureScreener.run()
        screening_log: str, optional
            Path to screening log file
        all_screening_results: dict, optional
            Complete screening results dictionary {pdf_file: {strufile: {'Rw': float, 'zoomscale': float}}}
            If provided, will be used instead of parsing the log file
        """
        self.strufile_dir = strufile_dir
        self.best_results = best_results
        self.generation_log = os.path.join(strufile_dir, 'structure_generation.log')
        self.screening_log = screening_log or os.path.join(strufile_dir, 'structure_screening.log')
        self.all_screening_results = all_screening_results
        
    def parse_generation_log(self):
        """
        Parse structure_generation.log to extract diameter, natoms, surface atoms info
        
        Returns:
        --------
        dict: {structure_filename: {'diameter': float, 'natoms': int, 'surface_atoms': int}}
        """
        structure_info = {}
        
        if not os.path.exists(self.generation_log):
            print(f"Warning: Generation log not found at {self.generation_log}")
            return structure_info
        
        with open(self.generation_log, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines
        data_started = False
        for line in lines:
            if 'Structure File' in line and 'Diameter' in line:
                data_started = True
                continue
            
            if data_started and line.strip():
                try:
                    parts = line.split()
                    if len(parts) >= 4:
                        filename = parts[0]
                        diameter = float(parts[1])
                        natoms = int(parts[2])
                        surface_atoms = int(parts[3])
                        
                        structure_info[filename] = {
                            'diameter': diameter,
                            'natoms': natoms,
                            'surface_atoms': surface_atoms,
                            'surface_fraction': surface_atoms / natoms if natoms > 0 else 0
                        }
                except (ValueError, IndexError):
                    continue
        
        return structure_info
    
    def generate_html_report(self, output_path='structure_screening_report.html'):
        """
        Generate comprehensive HTML report with structure visualizations
        
        Parameters:
        -----------
        output_path: str
            Path for output HTML file
        """
        structure_info = self.parse_generation_log()
        
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Structure Screening Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .best-result {
            background-color: #e8f5e9;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
        }
        .metric-label {
            font-weight: bold;
            color: #555;
        }
        .metric-value {
            color: #2c3e50;
            font-size: 1.2em;
        }
        .summary-box {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Structure Screening Report</h1>
        <p><strong>Generated:</strong> """ + str(np.datetime64('now')) + """</p>
        <p><strong>Structure Directory:</strong> """ + self.strufile_dir + """</p>
        
        <h2>🎯 Best Results Summary</h2>
"""
        
        # Add best results for each PDF
        for pdf_file, result in self.best_results.items():
            pdf_name = os.path.basename(pdf_file)
            strufile = result['strufile']
            strufile_basename = os.path.basename(strufile)
            rw = result['Rw']
            zoomscale = result['zoomscale']
            
            # Get structure info from generation log
            stru_info = structure_info.get(strufile_basename, {})
            
            html_content += f"""
        <div class="best-result">
            <h3>📄 {pdf_name}</h3>
            <div class="metric">
                <span class="metric-label">Structure:</span>
                <span class="metric-value">{strufile_basename}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Rw:</span>
                <span class="metric-value">{rw:.4f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Zoomscale:</span>
                <span class="metric-value">{zoomscale:.6f}</span>
            </div>
"""
            
            if stru_info:
                html_content += f"""
            <div class="metric">
                <span class="metric-label">Diameter:</span>
                <span class="metric-value">{stru_info.get('diameter', 'N/A'):.2f} Å</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Atoms:</span>
                <span class="metric-value">{stru_info.get('natoms', 'N/A')}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Surface Atoms:</span>
                <span class="metric-value">{stru_info.get('surface_atoms', 'N/A')} ({stru_info.get('surface_fraction', 0)*100:.1f}%)</span>
            </div>
"""
            
            html_content += """
        </div>
"""
        
        # Add structure info table
        if structure_info:
            html_content += """
        <h2>📋 All Generated Structures</h2>
        <table>
            <thead>
                <tr>
                    <th>Structure</th>
                    <th>Diameter (Å)</th>
                    <th>Total Atoms</th>
                    <th>Surface Atoms</th>
                    <th>Surface %</th>
                </tr>
            </thead>
            <tbody>
"""
            for filename, info in sorted(structure_info.items()):
                html_content += f"""
                <tr>
                    <td>{filename}</td>
                    <td>{info['diameter']:.2f}</td>
                    <td>{info['natoms']}</td>
                    <td>{info['surface_atoms']}</td>
                    <td>{info['surface_fraction']*100:.1f}%</td>
                </tr>
"""
            
            html_content += """
            </tbody>
        </table>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        # Write HTML file
        output_full_path = os.path.join(self.strufile_dir, output_path)
        with open(output_full_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Report generated: {output_full_path}")
        return output_full_path
    
    def generate_summary_dict(self):
        """
        Generate a comprehensive summary dictionary for programmatic access
        
        Returns:
        --------
        dict with complete information about screening results
        """
        structure_info = self.parse_generation_log()
        
        summary = {
            'strufile_dir': self.strufile_dir,
            'num_pdfs': len(self.best_results),
            'results': []
        }
        
        for pdf_file, result in self.best_results.items():
            strufile_basename = os.path.basename(result['strufile'])
            stru_info = structure_info.get(strufile_basename, {})
            
            result_dict = {
                'pdf_file': pdf_file,
                'pdf_name': os.path.basename(pdf_file),
                'strufile': result['strufile'],
                'strufile_name': strufile_basename,
                'Rw': result['Rw'],
                'zoomscale': result['zoomscale'],
                'diameter': stru_info.get('diameter'),
                'natoms': stru_info.get('natoms'),
                'surface_atoms': stru_info.get('surface_atoms'),
                'surface_fraction': stru_info.get('surface_fraction')
            }
            
            summary['results'].append(result_dict)
        
        return summary
    
    def send_report_by_email(self, report_path, recipient_email,  
                            smtp_server='smtp.gmail.com', smtp_port=587, subject=None, message=None):
        """
        Envoyer le rapport PDF par email
        
        Parameters:
        -----------
        report_path: str
            Chemin du fichier PDF à envoyer
        recipient_email: str or list
            Adresse(s) email du/des destinataire(s)
            Peut être une chaîne avec plusieurs emails séparés par des virgules
            ou une liste d'emails: ['email1@example.com', 'email2@example.com']
        sender_email: str
            Adresse email de l'expéditeur
        sender_password: str
            Mot de passe de l'email expéditeur (ou mot de passe d'application)
        smtp_server: str
            Serveur SMTP (défaut: smtp.gmail.com pour Gmail)
        smtp_port: int
            Port SMTP (défaut: 587 pour TLS)
        subject: str, optional
            Sujet de l'email (généré automatiquement si None)
        message: str, optional
            Corps du message (généré automatiquement si None)
            
        Returns:
        --------
        bool: True si envoyé avec succès, False sinon
        
        Notes:
        ------
        Pour Gmail, vous devez :
        1. Activer l'authentification à deux facteurs
        2. Générer un "mot de passe d'application" dans les paramètres de sécurité Google
        3. Utiliser ce mot de passe d'application comme sender_password
        
        Autres serveurs SMTP courants :
        - Outlook: smtp.office365.com, port 587
        - Yahoo: smtp.mail.yahoo.com, port 587
        - INSA: smtp.insa-rennes.fr, port 587 (ou selon config)
        """
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders
        
        try:
            # Vérifier que le fichier existe
            if not os.path.exists(report_path):
                print(f"❌ Erreur: Le fichier {report_path} n'existe pas")
             # Gérer plusieurs destinataires
            if isinstance(recipient_email, str):
                # Si c'est une chaîne, séparer par les virgules et nettoyer les espaces
                recipients = [email.strip() for email in recipient_email.split(',')]
            else:
                # Si c'est déjà une liste
                recipients = recipient_email
            
            # Créer le message
            msg = MIMEMultipart()
            msg['From'] = 'nicolas.ratel-ramond@insa-toulouse.fr'
            msg['To'] = ', '.join(recipients)  # Joindre les emails pour l'en-tête
            msg['From'] = 'nicolas.ratel-ramond@insa-toulouse.fr'
            msg['To'] = recipient_email
            
            # Sujet par défaut
            if subject is None:
                report_name = os.path.basename(report_path)
                subject = f"Structure Screening Report - {report_name}"
            msg['Subject'] = subject
            
            # Corps du message par défaut
            if message is None:
                summary = self.generate_summary_dict()
                message = f"""
Bonjour,

Veuillez trouver ci-joint le rapport d'analyse de structure.

Résumé des résultats :
- Nombre de PDF analysés : {summary['num_pdfs']}
- Répertoire : {summary['strufile_dir']}

"""
                for result in summary['results']:
                    message += f"\n{result['pdf_name']}:\n"
                    message += f"  • Best structure : {result['strufile_name']}\n"
                    message += f"  • Rw : {result['Rw']:.4f}\n"
                    message += f"  • Zoomscale : {result['zoomscale']:.6f}\n"
                    if result['natoms']:
                        message += f"  • Number of atoms : {result['natoms']}\n"
                        message += f"  • diameter : {result['diameter']:.2f} Å\n"
                
                
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Attacher le fichier PDF
            attachment = open(report_path, 'rb')
            part = MIMEBase('application', 'pdf')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(report_path)}")
            msg.attach(part)
            attachment.close()
            
            # Se connecter au serveur SMTP et envoyer
            print(f"📧 Connexion au serveur SMTP {smtp_server}:{smtp_port}...")
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()  # Sécuriser la connexion
            
            print(f"🔐 Authentification...")
            server.login('nicolas.ratel-ramond@insa-toulouse.fr', 'MNm12102012!')
            
            print(f"📤 Envoi du rapport à {', '.join(recipients)}...")
            text = msg.as_string()
            server.sendmail('nicolas.ratel-ramond@insa-toulouse.fr', recipients, text)  # recipients est une liste
            server.quit()
            
            print(f"✅ Email envoyé avec succès à {len(recipients)} destinataire(s)!")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de l'envoi de l'email: {e}")
            print(f"\nVérifiez:")
            print(f"  • Vos identifiants email")
            print(f"  • Que vous utilisez un mot de passe d'application (Gmail)")
            print(f"  • Votre connexion internet")
            print(f"  • Les paramètres SMTP de votre fournisseur")
            return False
    
    def parse_screening_log(self):
        """
        Parse structure_screening.log to extract all refinement results
        
        Returns:
        --------
        dict: {pdf_file: {strufile: {'Rw': float, 'zoomscale': float}}}
        """
        all_results = {}
        
        if not os.path.exists(self.screening_log):
            print(f"Warning: Screening log not found at {self.screening_log}")
            return all_results
        
        with open(self.screening_log, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines
        data_started = False
        for line in lines:
            # Skip header and separator lines
            if 'STRUCTURE SCREENING' in line or '*****' in line or 'PDF file' in line:
                if 'PDF file' in line and 'Structure file' in line:
                    data_started = True
                continue
            
            if not data_started:
                continue
            
            line = line.strip()
            if not line or line.startswith('Liste des meilleures') or line.startswith('Fichier PDF'):
                continue
            
            # Parse result lines: PDF_name \t Structure_name \t Rw \t zoomscale=value
            # or: PDF_name \t Structure_name \t Rw
            parts = [p.strip() for p in line.split('\t') if p.strip()]
            
            if len(parts) >= 3:
                try:
                    pdf_name = parts[0]
                    stru_name = parts[1]
                    
                    # Extract Rw and zoomscale from the third part
                    rw_zoomscale_str = parts[2]
                    
                    # Check if zoomscale is in a separate column or in the same
                    if len(parts) >= 4 and 'zoomscale=' in parts[3]:
                        # Format: PDF \t Structure \t Rw \t zoomscale=value
                        rw = float(rw_zoomscale_str)
                        zoomscale_str = parts[3].replace('zoomscale=', '').strip()
                        zoomscale = float(zoomscale_str)
                    elif 'zoomscale=' in rw_zoomscale_str:
                        # Format: PDF \t Structure \t Rw\tzoomscale=value
                        rw_part = rw_zoomscale_str.split('zoomscale=')[0].strip()
                        zoom_part = rw_zoomscale_str.split('zoomscale=')[1].strip()
                        rw = float(rw_part)
                        zoomscale = float(zoom_part)
                    else:
                        # Format: PDF \t Structure \t Rw (no zoomscale)
                        rw = float(rw_zoomscale_str)
                        zoomscale = None
                    
                    # Store results - use PDF basename if it's a full path
                    if '/' in pdf_name:
                        pdf_name = os.path.basename(pdf_name)
                    
                    if pdf_name and stru_name:
                        if pdf_name not in all_results:
                            all_results[pdf_name] = {}
                        all_results[pdf_name][stru_name] = {
                            'Rw': rw,
                            'zoomscale': zoomscale
                        }
                except (ValueError, IndexError) as e:
                    # Debug: print problematic line
                    # print(f"Could not parse line: {line} - Error: {e}")
                    continue
        
        return all_results
    
    def get_top_n_results(self, n=10, pdf_file=None):
        """
        Get top N refinement results sorted by Rw
        
        Parameters:
        -----------
        n: int
            Number of top results to return
        pdf_file: str, optional
            Specific PDF file to analyze. If None, uses first PDF in best_results
            
        Returns:
        --------
        list of dicts with structure info and Rw values
        """
        # Use provided all_screening_results or parse log
        if self.all_screening_results:
            all_results = self.all_screening_results
        else:
            all_results = self.parse_screening_log()
        
        if not all_results:
            print("No results found. Try providing all_screening_results when creating StructureReportGenerator.")
            return []
        
        # Select PDF file
        if pdf_file is None:
            # Get first PDF from best_results
            first_pdf = list(self.best_results.keys())[0]
            pdf_file = os.path.basename(first_pdf)
        else:
            pdf_file = os.path.basename(pdf_file)
        
        # Try different PDF name formats
        pdf_results = None
        if pdf_file in all_results:
            pdf_results = all_results[pdf_file]
        else:
            # Try without extension
            pdf_base = os.path.splitext(pdf_file)[0]
            for key in all_results.keys():
                if pdf_base in key or key in pdf_base:
                    pdf_results = all_results[key]
                    break
        
        if not pdf_results:
            print(f"PDF file {pdf_file} not found in results")
            print(f"Available PDFs: {list(all_results.keys())}")
            return []
        
        # Get structure info
        structure_info = self.parse_generation_log()
        
        # Collect and sort results
        results_list = []
        for stru_name, refinement in pdf_results.items():
            stru_info = structure_info.get(stru_name, {})
            
            result = {
                'structure_name': stru_name,
                'structure_path': os.path.join(self.strufile_dir, stru_name),
                'Rw': refinement['Rw'],
                'zoomscale': refinement.get('zoomscale', 1.0),
                'diameter': stru_info.get('diameter'),
                'natoms': stru_info.get('natoms'),
                'surface_atoms': stru_info.get('surface_atoms'),
                'surface_fraction': stru_info.get('surface_fraction')
            }
            results_list.append(result)
        
        # Sort by Rw and return top N
        results_list.sort(key=lambda x: x['Rw'])
        return results_list[:n]
    
    def generate_structure_thumbnail(self, strufile, zoomscale, output_path, size=(400, 400)):
        """
        Generate a thumbnail image of a structure using matplotlib
        
        Parameters:
        -----------
        strufile: str
            Path to structure file
        zoomscale: float
            Zoomscale to apply
        output_path: str
            Path where to save the thumbnail
        size: tuple
            Size of the image (width, height) in pixels
        """
        # Check if structure file exists
        if not os.path.exists(strufile):
            return None
        
        try:
            
            # Fallback to matplotlib if py3Dmol fails
            from matplotlib import pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from scipy.spatial import distance_matrix
            
            # Dictionnaires des propriétés atomiques
            atomic_radii = {
                'Au': 1.44, 'Ag': 1.45, 'Cu': 1.28, 'Pt': 1.39, 'Pd': 1.37,
                'Fe': 1.26, 'Ni': 1.24, 'Co': 1.25, 'Cr': 1.28, 'Mn': 1.27,
                'Ti': 1.47, 'V': 1.35, 'Zn': 1.34, 'Al': 1.43, 'Si': 1.18,
                'C': 0.77, 'O': 0.73, 'N': 0.71, 'H': 0.53, 'S': 1.04
            }
            
            element_colors = {
                'Au': 'gold', 'Ag': 'silver', 'Cu': 'orange', 'Pt': 'lightgray', 'Pd': 'lightblue',
                'Fe': 'orangered', 'Ni': 'lightgreen', 'Co': 'blue', 'Cr': 'gray', 'Mn': 'violet',
                'Ti': 'silver', 'V': 'darkgray', 'Zn': 'steelblue', 'Al': 'lightgray', 'Si': 'tan',
                'C': 'dimgray', 'O': 'red', 'N': 'blue', 'H': 'white', 'S': 'yellow'
            }
            
            data = np.loadtxt(strufile, skiprows=2, 
                             dtype=[('element', 'U2'), ('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
            coords = np.column_stack([data['x'], data['y'], data['z']]) * zoomscale
            
            # Détecter l'élément principal (le plus fréquent)
            elements, counts = np.unique(data['element'], return_counts=True)
            main_element = elements[np.argmax(counts)]
            main_element = main_element.strip()
            
            # Récupérer les propriétés de l'élément
            atom_radius = atomic_radii.get(main_element, 1.4)  # défaut si élément inconnu
            atom_color = element_colors.get(main_element, 'gray')
            edge_color = 'darkgray' if main_element not in ['Au', 'Ag'] else f'dark{atom_color}'
            
            # Calculer la distance minimale entre atomes voisins pour dimensionner les sphères
            if len(coords) > 1:
                dist_mat = distance_matrix(coords, coords)
                # Mettre la diagonale à inf pour ignorer la distance d'un atome à lui-même
                np.fill_diagonal(dist_mat, np.inf)
                min_dist = np.min(dist_mat)
                # Utiliser 85% de la distance minimale comme rayon pour un léger overlap
                sphere_radius = min_dist * 0.85 / 2.0
            else:
                sphere_radius = atom_radius
            
            fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            # Calculer la plage des axes
            max_range = np.array([coords[:, 0].max()-coords[:, 0].min(),
                                 coords[:, 1].max()-coords[:, 1].min(),
                                 coords[:, 2].max()-coords[:, 2].min()]).max() / 2.0
            
            mid_x = (coords[:, 0].max()+coords[:, 0].min()) * 0.5
            mid_y = (coords[:, 1].max()+coords[:, 1].min()) * 0.5
            mid_z = (coords[:, 2].max()+coords[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Convertir le rayon physique en taille de point pour scatter
            # La taille s dans scatter est en points^2
            # On calcule la taille en fonction de la plage de l'axe et de la résolution
            fig_size_inches = size[0] / 100  # taille de la figure en inches
            points_per_unit = (fig_size_inches * 72) / (2 * max_range)  # points par unité Å
            sphere_size_points = (sphere_radius * points_per_unit) ** 2
            
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                      c='gold', s=sphere_size_points, alpha=0.9, 
                      edgecolors='darkgoldenrod', linewidths=0.5)
            
            ax.set_xlabel('X (Å)', fontsize=8)
            ax.set_ylabel('Y (Å)', fontsize=8)
            ax.set_zlabel('Z (Å)', fontsize=8)
            ax.grid(False)
            ax.set_facecolor('white')
            # Masquer les axes pour un rendu plus propre
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close()
            return output_path
        except Exception:
            return None
    
    def generate_pdf_report(self, output_path='structure_screening_report.pdf', n_top=10, pdf_file=None):
        """
        Generate comprehensive PDF report with top N refinement results
        
        Parameters:
        -----------
        output_path: str
            Path for output PDF file
        n_top: int
            Number of top results to include (default: 10)
        pdf_file: str, optional
            Specific PDF file to analyze. If None, uses first PDF in best_results
        """
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            from matplotlib import pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("Matplotlib required for PDF generation")
            return None
        
        # Get top N results
        top_results = self.get_top_n_results(n=n_top, pdf_file=pdf_file)
        
        if not top_results:
            print("No results to generate report")
            return None
        
        # Select PDF file name
        if pdf_file is None:
            pdf_file = list(self.best_results.keys())[0]
        pdf_name = os.path.basename(pdf_file)
        
        # Create output directory for thumbnails
        thumb_dir = os.path.join(self.strufile_dir, 'thumbnails')
        os.makedirs(thumb_dir, exist_ok=True)
        
        # Full output path
        output_full_path = os.path.join(self.strufile_dir, output_path)
        
        with PdfPages(output_full_path) as pdf:
            # Page 1: Complete Overview - Fit Curve + Thumbnail + Details
            best_result = top_results[0]
            best_strufile_name = best_result['structure_name']
            
            # Get best structure file path
            if 'structure_path' in best_result and os.path.exists(best_result['structure_path']):
                best_strufile_path = best_result['structure_path']
            else:
                best_strufile_path = os.path.join(self.strufile_dir, best_strufile_name + '.xyz')
            
            # Generate thumbnail for best structure and save it
            best_thumb_path = os.path.join(thumb_dir, f"best_structure_{best_strufile_name}.png")
            
            if os.path.exists(best_strufile_path):
                self.generate_structure_thumbnail(
                    best_strufile_path, 
                    best_result['zoomscale'] if best_result['zoomscale'] else 1.0,
                    best_thumb_path,
                    size=(500, 500)
                )
            
            # Create comprehensive first page
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('Structure Screening Report - Best Result', fontsize=18, fontweight='bold', y=0.98)
            
            # Create grid for layout: [fit curve (top), thumbnail + info (bottom)]
            gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1.2, 1], 
                                  hspace=0.3, wspace=0.3, left=0.08, right=0.95, top=0.93, bottom=0.05)
            
            # Top: Fit curve (spans both columns)
            ax_fit = fig.add_subplot(gs[0, :])
            
            # Find fit data
            pdf_basename = os.path.basename(pdf_file).replace('.gr', '')
            png_locations = [
                os.path.join(self.strufile_dir, 'fig', f"{pdf_basename}_{best_strufile_name}.png"),
                os.path.join(self.strufile_dir, f"{pdf_basename}_{best_strufile_name}.png")
            ]
            fit_locations = [
                os.path.join(self.strufile_dir, 'fit', f"{pdf_basename}_{best_strufile_name}.fit"),
                os.path.join(self.strufile_dir, f"{pdf_basename}_{best_strufile_name}.fit")
            ]
            
            png_file = None
            for png_loc in png_locations:
                if os.path.exists(png_loc):
                    png_file = png_loc
                    break
            
            fit_file = None
            for fit_loc in fit_locations:
                if os.path.exists(fit_loc):
                    fit_file = fit_loc
                    break
            
            if png_file:
                img = plt.imread(png_file)
                ax_fit.imshow(img)
                ax_fit.axis('off')
                ax_fit.set_title(f'Best Fit: {best_strufile_name} (Rw={best_result["Rw"]:.4f})', 
                                fontsize=12, fontweight='bold')
            elif fit_file:
                data = np.loadtxt(fit_file, skiprows=0)
                r = data[:, 0]
                g_obs = data[:, 1]
                g_calc = data[:, 2]
                diff = g_obs - g_calc
                
                ax_fit.plot(r, g_obs, 'bo', label='Observed', markersize=2, alpha=0.6)
                ax_fit.plot(r, g_calc, 'r-', label='Calculated', linewidth=1.5)
                ax_fit.set_ylabel('G(r) (Å⁻²)', fontsize=10)
                ax_fit.set_xlabel('r (Å)', fontsize=10)
                ax_fit.set_title(f'Best Fit: {best_strufile_name} (Rw={best_result["Rw"]:.4f})', 
                                fontsize=12, fontweight='bold')
                ax_fit.legend(fontsize=8)
                ax_fit.grid(alpha=0.3)
            else:
                ax_fit.text(0.5, 0.5, 'Fit curve not available', 
                           ha='center', va='center', fontsize=10)
                ax_fit.axis('off')
            
            # Bottom left: Structure thumbnail
            ax_thumb = fig.add_subplot(gs[1, 0])
            if os.path.exists(best_thumb_path):
                thumb_img = plt.imread(best_thumb_path)
                ax_thumb.imshow(thumb_img)
                ax_thumb.set_title('3D Structure', fontsize=11, fontweight='bold')
            else:
                ax_thumb.text(0.5, 0.5, 'Thumbnail\nnot available', 
                             ha='center', va='center', fontsize=10)
            ax_thumb.axis('off')
            
            # Bottom right: Detailed information
            ax_info = fig.add_subplot(gs[1, 1])
            ax_info.axis('off')
            
            # Format values
            best_zoomscale = f"{best_result['zoomscale']:.6f}" if best_result['zoomscale'] else 'N/A'
            best_diameter = f"{best_result['diameter']:.2f}" if best_result['diameter'] else 'N/A'
            best_natoms = best_result['natoms'] if best_result['natoms'] else 'N/A'
            best_surface = best_result['surface_atoms'] if best_result['surface_atoms'] else 'N/A'
            best_surf_pct = f"({best_result['surface_fraction']*100:.1f}%)" if best_result['surface_fraction'] else ''
            
            info_text = f"""
PDF: {pdf_name}

═══════════════════════
REFINEMENT RESULTS:
═══════════════════════
Rw: {best_result['Rw']:.4f}
Zoomscale: {best_zoomscale}

═══════════════════════
STRUCTURE PROPERTIES:
═══════════════════════
Name: {best_strufile_name}
Diameter: {best_diameter} Å
Total atoms: {best_natoms}
Surface atoms: {best_surface} {best_surf_pct}

═══════════════════════
REPORT INFO:
═══════════════════════
Generated: {np.datetime64('now')}
Top {n_top} results included
            """
            
            ax_info.text(0.05, 0.95, info_text, fontsize=9, family='monospace',
                        verticalalignment='top', transform=ax_info.transAxes)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2-3: Top N Results Table
            n_per_page = 15
            for page_num, i in enumerate(range(0, len(top_results), n_per_page)):
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                
                page_results = top_results[i:i+n_per_page]
                
                # Create table data
                table_data = [['Rank', 'Structure', 'Rw', 'Zoomscale', 'Diam.(Å)', 'Atoms', 'Surf.%']]
                
                for idx, res in enumerate(page_results, start=i+1):
                    row = [
                        f'{idx}',
                        res['structure_name'][:30],
                        f"{res['Rw']:.4f}",
                        f"{res['zoomscale']:.4f}" if res['zoomscale'] else 'N/A',
                        f"{res['diameter']:.1f}" if res['diameter'] else 'N/A',
                        f"{res['natoms']}" if res['natoms'] else 'N/A',
                        f"{res['surface_fraction']*100:.1f}" if res['surface_fraction'] else 'N/A'
                    ]
                    table_data.append(row)
                
                table = ax.table(cellText=table_data, loc='center', cellLoc='left')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2)
                
                # Style header row
                for i in range(len(table_data[0])):
                    table[(0, i)].set_facecolor('#3498db')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                # Alternate row colors
                for i in range(1, len(table_data)):
                    for j in range(len(table_data[0])):
                        if i % 2 == 0:
                            table[(i, j)].set_facecolor('#f0f0f0')
                
                ax.set_title(f'Top {n_top} Refinement Results (Page {page_num+1})', 
                           fontsize=14, fontweight='bold', pad=20)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Page: Structure Thumbnails (4 per page)
            for page_idx in range(0, min(n_top, 12), 4):
                fig, axes = plt.subplots(2, 2, figsize=(8.5, 11))
                fig.suptitle(f'Structure Visualizations (Rank {page_idx+1}-{page_idx+4})', 
                           fontsize=14, fontweight='bold')
                
                axes = axes.flatten()
                
                for idx in range(4):
                    if page_idx + idx >= len(top_results):
                        axes[idx].axis('off')
                        continue
                    
                    res = top_results[page_idx + idx]
                    
                    # Use structure_path if available, otherwise construct it
                    if 'structure_path' in res and os.path.exists(res['structure_path']):
                        strufile_path = res['structure_path']
                    else:
                        strufile_path = os.path.join(self.strufile_dir, res['structure_name'] + '.xyz')
                    
                    thumb_path = os.path.join(thumb_dir, f"thumb_{page_idx+idx}.png")
                    
                    if os.path.exists(strufile_path):
                        self.generate_structure_thumbnail(
                            strufile_path, 
                            res['zoomscale'] if res['zoomscale'] else 1.0,
                            thumb_path
                        )
                        
                        if os.path.exists(thumb_path):
                            img = plt.imread(thumb_path)
                            axes[idx].imshow(img)
                    
                    axes[idx].axis('off')
                    title_text = f"#{page_idx+idx+1}: Rw={res['Rw']:.4f}\n{res['natoms']} atoms" if res['natoms'] else f"#{page_idx+idx+1}: Rw={res['Rw']:.4f}"
                    axes[idx].set_title(title_text, fontsize=10)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        return output_full_path




class PDFRefinement():
    def __init__(self,
                 pdffile:str,
                 strufile:str,
                 qdamp:float=0.014,
                 qbroad:float=0.04,
                 refinement_tags:dict={'scale_factor': True, 'zoomscale': True, 'delta2': True, 'Uiso': True},
                 save_tag:bool=False,
                 RUN_PARALLEL:bool=True,
                 rmin=0.01,
                 rbins:int=1,
                 screening_tag:bool=False):

                 """
                 refinement_tags={'scale_factor': True, 'zoomscale': True, 'delta2': True, 'Uiso': True}
                 pdffile: path to pdf file
                 strufile path to structure file
                 qdamp qdamp value (default=0.014)
                 qbroad qbroad value (default==0.04)
                 save_tag: save refinement data (default=False)
                 RUN_PARALLEL=True
                 rbins: int, can be adjusted to increase rstep (default=1)
                 screening_tag=False
                 """
                # Check file formats
                 pdf_extension=os.path.basename(pdffile).split('.')[-1]
                 if pdf_extension == 'gr':
                     self.pdffile = pdffile
                 else:
                     print('PDF file should be a .gr file, extracted with pdfgtetx3')
                 stru_extension=os.path.basename(strufile).split('.')[-1]
                 if stru_extension == 'xyz':
                     self.strufile = strufile
                 else:
                     print('Structure files must adopt the xyz standard format')

                # Initialize attributes
                 self.path=os.path.dirname(self.strufile)
                 self.qdamp = qdamp
                 self.qbroad = qbroad
                 self.refinement_tags = refinement_tags
                 self.save_tag = save_tag
                 self.RUN_PARALLEL=RUN_PARALLEL
                 self.rbins=rbins
                 self.screening_tag=screening_tag
                 # Read metadata from pdffile
                 with open(self.pdffile, 'r') as f:
                     for line in f:
                         if "qmin" in line:
                             self.qmin = float(line.split(' = ')[1].strip())
                         if "qmax" in line:
                             self.qmax = float(line.split(' = ')[1].strip())
                 # Load data from the PDF file
                 r = np.loadtxt(self.pdffile, usecols=(0), skiprows=29)
                 self.rmin = rmin
                 self.rmax = np.max(r) 
                 self.rstep = ((self.rmax-self.rmin) / (len(r) - 1))*self.rbins

                 # Create fit recipe
                 self.recipe = self.make_recipe()

    def file_extension(self, file):
        return os.path.basename(file).split('.')[-1]
    
    def make_recipe(self):
        PDF_RMIN=self.rmin
        PDF_RMAX=self.rmax
        PDF_RSTEP=self.rstep
        QBROAD_I=self.qbroad
        QDAMP_I=self.qdamp
        QMIN=self.qmin
        QMAX=self.qmax
        ZOOMSCALE_I=1
        UISO_I=0.005
        stru1 = Structure(filename=self.strufile)

        profile = Profile()
        parser = PDFParser()
        parser.parseFile(self.pdffile)
        profile.loadParsedData(parser)
        profile.setCalculationRange(xmin=PDF_RMIN, xmax=PDF_RMAX, dx=PDF_RSTEP)

        # 10: Create a Debye PDF Generator object for the discrete structure model.
        generator_cluster1 = DebyePDFGenerator("G1")
        generator_cluster1.setStructure(stru1, periodic=False)

        # 11: Create a Fit Contribution object.
        contribution = FitContribution("cluster")
        contribution.addProfileGenerator(generator_cluster1)
                
        # If you have a multi-core computer (you probably do), run your refinement in parallel!
        if self.RUN_PARALLEL:
            try:
                import psutil
                import multiprocessing
                from multiprocessing import Pool
            except ImportError:
                print("\nYou don't appear to have the necessary packages for parallelization")
            syst_cores = multiprocessing.cpu_count()
            cpu_percent = psutil.cpu_percent()
            avail_cores = np.floor((100 - cpu_percent) / (100.0 / syst_cores))
            ncpu = int(np.max([1, avail_cores]))
            pool = Pool(processes=ncpu)
            generator_cluster1.parallel(ncpu=ncpu, mapfunc=pool.map)
            
        contribution.setProfile(profile, xname="r")

        # 13: Set an equation, based on your PDF generators. 
        contribution.setEquation("s1*G1")

        # 14: Create the Fit Recipe object that holds all the details of the fit.
        recipe = FitRecipe()
        recipe.addContribution(contribution)

        # 15: Initialize the instrument parameters, Q_damp and Q_broad, and
        # assign Q_max and Q_min.
        generator_cluster1.qdamp.value = QDAMP_I
        generator_cluster1.qbroad.value = QBROAD_I
        generator_cluster1.setQmax(QMAX)
        generator_cluster1.setQmin(QMIN)

        # 16: Add, initialize, and tag variables in the Fit Recipe object.
        # In this case we also add psize, which is the NP size.
        recipe.addVar(contribution.s1, float(1), tag="scale_factor")

        # 17: Define a phase and lattice from the Debye PDF Generator
        # object and assign an isotropic lattice expansion factor tagged
        # "zoomscale" to the structure. 
        phase_cluster1 = generator_cluster1.phase
        lattice1 = phase_cluster1.getLattice()
        recipe.newVar("zoomscale", ZOOMSCALE_I, tag="zoomscale")
        recipe.constrain(lattice1.a, 'zoomscale')
        recipe.constrain(lattice1.b, 'zoomscale')
        recipe.constrain(lattice1.c, 'zoomscale')
        # 18: Initialize an atoms object and constrain the isotropic
        # Atomic Displacement Paramaters (ADPs) per element. 
        atoms1 = phase_cluster1.getScatterers()
        recipe.newVar("Uiso", UISO_I, tag="Uiso")
        for atom in atoms1:
            recipe.constrain(atom.Uiso, "Uiso")
            recipe.restrain("Uiso",lb=0,ub=1,scaled=True,sig=0.00001)
        recipe.addVar(generator_cluster1.delta2, name="delta2", value=float(4), tag="delta2")
        recipe.restrain("delta2",lb=0,ub=12,scaled=True,sig=0.00001)
        return recipe
    
       
    def get_filename(self,file):
        filename=os.path.basename(file).split('/')[-1]
        return filename.split('.')[0]

    def refine(self):
        # Establish the location of the data and a name for our fit.
        gr_path = str(self.pdffile)
        FIT_ID=self.get_filename(self.pdffile)+'_'+self.get_filename(self.strufile)
        basename = FIT_ID        
        # Establish the full path of the structure file
        stru_path = self.strufile
        recipe = self.recipe
        # Amount of information to write to the terminal during fitting.
        if not self.screening_tag:
            recipe.fithooks[0].verbose = 3
        else:
            recipe.fithooks[0].verbose = 0


        recipe.fix("all")
        # Define values to refin from self.refinement_tags
        tags=[]
        for key in self.refinement_tags: 
            if self.refinement_tags[key]==True:
                tags.append(key)
        
        tags.append("all")
        for tag in tags:
            recipe.free(tag)
            
            least_squares(recipe.residual, recipe.values, x_scale="jac")

        # Write the fitted data to a file.
        profile = recipe.cluster.profile
        #profile.savetxt(fitdir / f"{basename}.fit")

        res = FitResults(recipe)
        if not self.screening_tag:
            res.printResults()
        
        #res.saveResults(resdir / f"{basename}.res", header=header)

        # Save refinement results        
        if self.save_tag:
            self.save_fitresults(profile,res)
        else: 
            pass
        return res.rw
    
    def save_fitresults(self,profile,res):
        basename=self.get_filename(self.pdffile)+'_'+self.get_filename(self.strufile)
        
        PWD=Path(self.path)
        # Make some folders to store our output files.
        resdir = PWD / "res"
        fitdir = PWD / "fit"
        figdir = PWD / "fig"
        folders = [resdir, fitdir, figdir]
        for folder in folders:
            if not folder.exists():
                folder.mkdir()
        # save exp and calc pdf
        profile.savetxt(fitdir / f"{basename}.fit")
        # Write the fit results to a file.
        header = "%s"%str(basename)+".\n"
        header+="data file:%s"%str(self.pdffile)+"\n"
        header+="structure file:%s"%str(self.strufile)+"\n"
        header+="Fitting parameters \n"
        header+="rmin=%f"%self.rmin+"\n"
        header+="rmax=%f"%self.rmax+"\n"
        header+="rstep=%f"%self.rstep+"\n"
        header+="QBROAD=%f"%self.qbroad+"\n"
        header+="QDAMP=%f"%self.qdamp+"\n"
        header+="QMIN=%f"%self.qmin+"\n"
        header+="QMAX=%f"%self.qmax+"\n"
        res.saveResults(resdir / f"{basename}.res", header=header)

        #Make plot
        fig_name= figdir / basename
        if not isinstance(fig_name, Path):
            fig_name = Path(fig_name)
        plt.clf()
        plt.close('all')
        r = self.recipe.cluster.profile.x
        g = self.recipe.cluster.profile.y
        gcalc = self.recipe.cluster.profile.ycalc
        # Make an array of identical shape as g which is offset from g.
        diff = g - gcalc
        diffzero = (min(g)-np.abs(max(diff))) * \
            np.ones_like(g)
        # Calculate the residual (difference) array and offset it vertically.
        diff = g - gcalc + diffzero
        # Change some style details of the plot
        mpl.rcParams.update(mpl.rcParamsDefault)
        # Create a figure and an axis on which to plot
        fig, ax1 = plt.subplots(1, 1)
        # Plot the difference offset line
        ax1.plot(r, diffzero, lw=1.0, ls="--", c="black")
        # Plot the measured data
        ax1.plot(r,g,ls="None",marker="o",ms=5,mew=0.2,mfc="None",label="G(r) Data")
        ax1.plot(r, diff, lw=1.2, label="G(r) diff")
        ax1.plot(r,gcalc,'g',label='G(r) calc')
        ax1.set_xlabel(r"r ($\mathrm{\AA}$)")
        ax1.set_ylabel(r"G ($\mathrm{\AA}$$^{-2}$)")
        ax1.tick_params(axis="both",which="major",top=True,right=True)
        ax1.set_xlim(self.rmin, self.rmax)
        ax1.legend(ncol=2)
        fig.tight_layout()
        ax1.set_title(basename+'\n'+f'Rw={res.rw:.4f}')
        # Save plot
        fig.savefig(fig_name.parent / f"{fig_name.name}.png", format="png")


class PDFRefinementFast:
    """
    Fast PDF refinement class for STRUCTURE SCREENING.
    Same interface as PDFRefinement, but MUCH faster.
    """

    def __init__(self,
                 pdffile: str,
                 strufile: str,
                 qdamp: float = 0.014,
                 qbroad: float = 0.04,
                 rbins: int = 4,
                 rmin: float = 2.0,
                 rmax_fast: float = 15.0,
                 screening_tag: bool = True):

        self.pdffile = pdffile
        self.strufile = strufile
        self.qdamp = qdamp
        self.qbroad = qbroad
        self.rbins = rbins
        self.rmin = rmin
        self.rmax_fast = rmax_fast
        self.screening_tag = screening_tag

        self.recipe = self._make_fast_recipe()

    # ------------------------------------------------------------

    def _make_fast_recipe(self):
        import numpy as np
        from diffpy.srfit.fitbase import FitRecipe, FitContribution, Profile
        from diffpy.srfit.pdf import PDFParser, DebyePDFGenerator
        from diffpy.structure import Structure

        # --- Structure
        stru = Structure(filename=self.strufile)

        # --- PDF data
        profile = Profile()
        parser = PDFParser()
        parser.parseFile(self.pdffile)
        profile.loadParsedData(parser)

        r = profile.x
        rmax_data = np.max(r)
        rmax = min(self.rmax_fast, rmax_data)

        # Coarsen grid (rbins)
        rstep = (rmax - self.rmin) / (len(r) // self.rbins)

        profile.setCalculationRange(
            xmin=self.rmin,
            xmax=rmax,
            dx=rstep
        )

        # --- Debye generator
        gen = DebyePDFGenerator("G")
        gen.setStructure(stru, periodic=False)
        gen.qdamp.value = self.qdamp
        gen.qbroad.value = self.qbroad

        # --- Contribution
        contrib = FitContribution("cluster")
        contrib.addProfileGenerator(gen)
        contrib.setProfile(profile, xname="r")
        contrib.setEquation("s*G")

        # --- Recipe
        recipe = FitRecipe()
        recipe.addContribution(contrib)

        # --- Minimal parameter set
        recipe.addVar(contrib.s, 1.0, tag="scale")

        phase = gen.phase
        lattice = phase.getLattice()

        recipe.newVar("zoomscale", 1.0, tag="zoomscale")
        recipe.constrain(lattice.a, "zoomscale")
        recipe.constrain(lattice.b, "zoomscale")
        recipe.constrain(lattice.c, "zoomscale")

        # Fix everything except scale + zoomscale
        recipe.fix("all")
        recipe.free("scale")
        recipe.free("zoomscale")

        # Silence output
        recipe.fithooks[0].verbose = 0

        return recipe

    # ------------------------------------------------------------

    def refine(self):
        from scipy.optimize import least_squares
        from diffpy.srfit.fitbase import FitResults

        least_squares(
            self.recipe.residual,
            self.recipe.values,
            x_scale="jac",
            max_nfev=12
        )

        res = FitResults(self.recipe)
        return res.rw




class StructureScreener():
    
    def __init__(self,
                 strufile_dir:str,
                 pdffile_dir:str,
                 qdamp:float =0.014,
                 qbroad:float =0.04,
                 refinement_tags: dict ={'scale_factor': True, 'zoomscale': True, 'delta2': True, 'Uiso': True},
                 save_tag: bool=False,
                 RUN_PARALLEL:bool =True,
                 rbins : int =1,
                 rmin=0.01,
                 rmax_fast=15.0,

                 fast_screening: bool =False,
                 candidate_list: dict =None,
                 threshold_percent: float =5.0):
                 """
                 strufile_dir: path of directory containing structure files
                 pdffile_dir: path of directory containing pdf files
                 refinement_tags: dict ={'scale_factor': True, 'zoomscale': True, 'delta2': True, 'Uiso': True}
                 qdamp:float =0.014
                 qbroad:float =0.04
                 save_tag: bool=False
                 RUN_PARALLEL:bool =True
                 rbins : int =1
                 screening_tag: bool =True
                 candidate_list: dict = None (pass short-list from first screening for refinement)
                 threshold_percent: float = 5.0 (tolerance for candidate selection: min(Rw) ± threshold_percent%)
                 """       
                 self.strufile_dir=strufile_dir
                 self.pdffile_dir=pdffile_dir
                 self.qdamp=qdamp
                 self.qbroad=qbroad
                 self.refinement_tags=refinement_tags
                 self.save_tag=save_tag
                 self.RUN_PARALLEL=RUN_PARALLEL
                 self.rbins=rbins
                 self.rmin = rmin
                 self.screening_tag=True
                 self.logfile=self.strufile_dir+'/structure_screening.log'
                 self.fast_screening=fast_screening
                 self.candidate_list=candidate_list
                 self.threshold_percent=threshold_percent
                 self.rmax_fast=rmax_fast
        
    def get_filename(self,file):
        filename=os.path.basename(file).split('/')[-1]
        return filename.split('.')[0]
    
    def extract_phi(self,filename):
        match = re.search(r'_phi=(\d+)', filename)
    
        # Return the extracted number as an integer
        return int(match.group(1))
    
        

    def run(self):
        """
        PDF refinement of each PDF file in pdffile_dir with each structure file in strufile_dir
        Returns:
            - If fast_screening=True: (best_results, candidate_list) tuple
            - If fast_screening=False: best_results dict only
        """
        from tqdm import tqdm
        
        best_results={}
        candidate_list = {}
        pdffile_list=glob.glob(os.path.join(self.pdffile_dir,'*.gr'))

        # Get structure list based on screening type
        if self.candidate_list is None:  # First screening (fast or full)
            strufile_list=glob.glob(os.path.join(self.strufile_dir,'*.xyz'))
            strufile_list=sorted(strufile_list,key=self.extract_phi)
        else:  # Second screening with candidate_list
            # Get all unique structures from candidate_list
            all_strufiles = set()
            for pdf_structures in self.candidate_list.values():
                all_strufiles.update(pdf_structures)
            strufile_list = sorted(list(all_strufiles), key=self.extract_phi)
        
        # Check if generator has kept_structures attribute (from auto mode)
        # If yes, use only those structures
        kept_structures_file = os.path.join(self.strufile_dir, 'kept_structures.txt')
        if os.path.exists(kept_structures_file) and self.candidate_list is None:
            with open(kept_structures_file, 'r') as f:
                kept_list = [line.strip() for line in f if line.strip()]
            if kept_list:
                strufile_list = kept_list
                print(f"Using {len(strufile_list)} structures in target diameter range")
        
        line2write= '*****************************************************\n\n'
        line2write+='                 STRUCTURE SCREENING                 \n\n'
        line2write+='*****************************************************\n\n'
        line2write+=f'PDF file       \tStructure file                                   \tRw\n\n'
        j=0
        print(line2write)
        print(f"Number of PDF files to process: {len(pdffile_list)}")
        
        # Calculate total number of refinements for progress bar AFTER determining structures to use
        total_refinements = 0
        for pdffile in pdffile_list:
            if self.candidate_list is not None:
                pdf_key = os.path.basename(pdffile)
                if pdf_key in self.candidate_list:
                    total_refinements += len(self.candidate_list[pdf_key])
            else:
                # Count only structures that will actually be tested
                total_refinements += len(strufile_list)
        
        # Single progress bar for all refinements
        pbar = tqdm(total=total_refinements, desc="Refining structures", ncols=80)
        
        refinement_count = 0  # Track actual refinements
        
        for pdffile in pdffile_list:
            pdfname=self.get_filename(pdffile)
            
            # Determine which structures to test for this PDF
            if self.candidate_list is not None:
                # Use only candidates for this PDF
                pdf_key = os.path.basename(pdffile)
                if pdf_key not in self.candidate_list:
                    j += 1
                    continue
                strufile_list_to_use = self.candidate_list[pdf_key]
            else:
                # Use all structures (already filtered by kept_structures.txt if available)
                strufile_list_to_use = strufile_list
            
            # Store refinement results (Rw and zoomscale) for this PDF
            refinement_results = {}
            
            for strufile in strufile_list_to_use:
                struname=self.get_filename(strufile)
                if self.fast_screening:
                    calc = PDFRefinementFast(
                        pdffile,
                        strufile,
                        rbins=self.rbins,
                        rmin=self.rmin,
                        rmax_fast=self.rmax_fast
                    )
                else:
                    calc=PDFRefinement(pdffile,
                                    strufile,
                                    refinement_tags=self.refinement_tags,
                                    save_tag=self.save_tag,
                                    rbins=self.rbins,
                                    rmin = self.rmin,
                                    screening_tag=self.screening_tag)
                rw=calc.refine()
                # Extract zoomscale from recipe
                zoomscale = calc.recipe.zoomscale.value
                refinement_results[strufile] = {'Rw': rw, 'zoomscale': zoomscale}
                temp=f'{pdfname:15}\t{struname:50}\t{rw:.4f}\tzoomscale={zoomscale:.6f}'
                print(temp)
                line2write+=f'{pdfname:15}\t{struname:50}\t{rw:.4f}\tzoomscale={zoomscale:.6f}\n'
                refinement_count += 1
                pbar.update(1)  # Update progress bar after each refinement
            
            # Only compute candidate list if not already provided
            if self.candidate_list is None:
                # the following code is to extract structures with Min(Rwp) +- threshold%
                min_rw = min(result['Rw'] for result in refinement_results.values())
                threshold_low = min_rw * (1 - self.threshold_percent/100.0)
                threshold_high = min_rw * (1 + self.threshold_percent/100.0)

                pdfname_full = os.path.basename(pdffile)

                best_results_candidates = {}
                best_results_candidates[pdfname_full] = {}

                for strufile, result in refinement_results.items():
                    rw = result['Rw']
                    if threshold_low <= rw <= threshold_high:
                        best_results_candidates[pdfname_full][strufile] = result

                # Affichage trié par Rw croissant
                print("****************************************************\nListe des meilleures structures candidates (min(R_w) ± "+str(self.threshold_percent)+"%) :\n")
                line2write += '*******************************************************\nListe des meilleures structures candidates (min(R_w) ± '+str(self.threshold_percent)+'%) :\n'

                for key, struct_dict in best_results_candidates.items():
                    # Trier par Rw croissant
                    sorted_items = sorted(struct_dict.items(), key=lambda item: item[1]['Rw'])
                    candidate_list[key] = [item[0] for item in sorted_items]  # Store sorted structure paths
                    
                    for file, result in sorted_items:
                        print(f'Fichier PDF : {key}, Structure : {self.get_filename(file)}, Rw = {result["Rw"]:.4f}, zoomscale = {result["zoomscale"]:.6f}\n')
                        line2write += f'Fichier PDF : {key}, Structure : {self.get_filename(file)}, Rw = {result["Rw"]:.4f}, zoomscale = {result["zoomscale"]:.6f}\n'
            
            # Find best results (minimum Rw)
            best_strufile_item = min(refinement_results.items(), key=lambda x: x[1]['Rw'])
            best_strufile = best_strufile_item[0]
            best_result = best_strufile_item[1]
            best_rw = best_result['Rw']
            best_zoomscale = best_result['zoomscale']
            pdfname=os.path.basename(pdffile)
            beststru=os.path.basename(best_strufile)
            best_results[pdffile]={'strufile': best_strufile, 'Rw': best_rw, 'zoomscale': best_zoomscale}
            line2write+='*******************************************************\n'
            line2write+=f'{pdfname}\t best structure={beststru} \t Rw={best_rw:.4f}\t zoomscale={best_zoomscale:.6f}\n\n'
            print("****************************************************\n")
            print(f'{pdfname}\t best structure={beststru} \t Rw={best_rw:.4f}\t zoomscale={best_zoomscale:.6f}\n')
            j+=1
        
        pbar.close()  # Close progress bar
        
        with open(self.logfile,'w') as f:
            f.write(line2write)
        
        # Return candidate_list if fast_screening (for use in refinement pass)
        if self.fast_screening and self.candidate_list is None:
            return best_results, candidate_list
        else:
            return best_results
        
                 

        

        




    
    
    
 
