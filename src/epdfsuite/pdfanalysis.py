from PDF_BatchAnalysis import StructureScreener, StructureGenerator, PDFRefinement, StructureCustom, StructureReportGenerator
import numpy as np
import glob
import os
from matplotlib import pyplot as plt
import json


def perform_automatic_pdf_analysis(
        pdf_file,
        # structure generation parameters
        cif_file,
        r_coh = None,
        tolerance_size_structure=3,
        n_spheres=2,
        max_search_param=25,
        # fast screening parameters
        rbins_fast=5,
        rmin=2.0,
        rmax_fast=15.0,
        threshold_percent_fast=5,
        # final refinement parameters
        rbins_fine = 1):
    
    from IPython.display import clear_output, display, HTML
    import shutil
    import tempfile
    
    path = os.path.dirname(pdf_file)
    r_exp, g = np.loadtxt(pdf_file, unpack=True, skiprows=27)
    
    # Create temporary directory containing only the specified PDF file
    # This ensures StructureScreener only processes this single file
    temp_pdf_dir = tempfile.mkdtemp(prefix='pdf_analysis_')
    temp_pdf_path = os.path.join(temp_pdf_dir, os.path.basename(pdf_file))
    shutil.copy2(pdf_file, temp_pdf_path)
    
    # Determine r_coh automatically if not provided
    if r_coh is None:
        clear_output(wait=True)
        display(HTML("<h3>🔍 STEP 0: Automatic determination of r_coh...</h3>"))
        print(f"PDF file: {os.path.basename(pdf_file)}")
        print("No r_coh provided, determining automatically from PDF...")
        print("⚠️  WARNING: Automatic detection can be unreliable for some systems.")
        print("   If results are poor, please provide r_coh manually after visual PDF inspection.")
        
        # Create temporary StructureGenerator to access analyze_pdf_and_get_rmax
        temp_gen = StructureGenerator(
            pdfpath=path,
            cif_file=cif_file,
            auto_mode=True,
            pdf_file=pdf_file,
            derivative_sigma=3.0,
            amplitude_sigma=2.0,
            window_size=20
        )
        r_coh = temp_gen.analyze_pdf_and_get_rmax()
        print(f"✓ Automatically determined r_coh: {r_coh:.2f} Å")
        
        # Sanity check: if r_coh is suspiciously small, warn user
        if r_coh < 15:
            print(f"⚠️  WARNING: Detected r_coh = {r_coh:.2f} Å seems quite small!")
            print("   This may indicate early detection of a local minimum.")
            print("   Consider providing r_coh manually if results are unsatisfactory.")
    
    # Étape 1 : Génération de structures candidates
    clear_output(wait=True)
    display(HTML("<h3>📁 STEP 1/4: Generating candidate structures within specified size range...</h3>"))
    print(f"PDF file: {os.path.basename(pdf_file)}")
    print(f"Coherence length: {r_coh:.2f} Å")
    print(f"Tolerance: ±{tolerance_size_structure:.2f}$\AA$")
    
    gen = StructureGenerator(
        pdfpath = path,
        cif_file=cif_file,
        auto_mode=True,
        pdf_file=pdf_file,
        r_coh=r_coh,
        tolerance=tolerance_size_structure,
        n_sizes=n_spheres,
        max_search_param=max_search_param,
        derivative_sigma=3.0,   # Soft weighting for derivative (used in continuous scoring)
        amplitude_sigma=2.0,     # Soft weighting for amplitude (used in continuous scoring)
        window_size=20,
        n_jobs=-1  # -1 = tous les CPU, ou spécifier un nombre
    )
    strufile_dir = gen.run()
    
    # Étape 2 : Screening rapide des structures candidates
    clear_output(wait=True)
    display(HTML("<h3>⚡ STEP 2/4: Fast screening of structures...</h3>"))
    print(f"Structure directory: {strufile_dir}")
    print(f"Fast screening parameters: rbins={rbins_fast}, rmax={rmax_fast} Å")
    
    screener_fast = StructureScreener(
        strufile_dir=strufile_dir,
        pdffile_dir=temp_pdf_dir,  # Use temporary directory with single PDF
        fast_screening=True,          # ⚡ Mode rapide
        rbins=rbins_fast,                       # Grid coarse
        rmin=rmin,
        rmax_fast=rmax_fast,
        threshold_percent=threshold_percent_fast )

    best_results_fast, candidate_list = screener_fast.run()
    
    # Afficher les candidats sélectionnés
    n_candidates = sum(len(structs) for structs in candidate_list.values())
    print(f"\n✓ Fast screening complete: {n_candidates} candidates selected")
    
    # Étape 3 : Affinement complet des meilleures structures candidates
    clear_output(wait=True)
    display(HTML("<h3>🔬 STEP 3/4: Fine refinement of best candidates...</h3>"))
    print(f"Refining {n_candidates} candidate structures")
    print(f"Refinement parameters: rbins={rbins_fine}, rmin={rmin:.1f} Å")
    
    screener_fine = StructureScreener(
        strufile_dir=strufile_dir,
        pdffile_dir=temp_pdf_dir,  # Use temporary directory with single PDF
        fast_screening=False,          # Affinement complet
        candidate_list=candidate_list, 
        refinement_tags={
            'scale_factor': True,
            'zoomscale': True,
            'delta2': True,
            'Uiso': True
        },
        save_tag=True,                # Sauvegarder les résultats détaillés
        rbins=rbins_fine,                       # Grille fine
        rmin=rmin,
        threshold_percent=0.0          # Critère de sélection
)
    best_results_fine = screener_fine.run()  

    # save reuslts to JSON
    with open(strufile_dir+"/screening_results.json", "w") as f:
        json.dump(best_results_fine, f)

    print(f"\n✓ Refinement complete")
    
    # Étape 4 : Génération du rapport et visualisation
    clear_output(wait=True)
    display(HTML("<h3>📊 STEP 4/4: Generating report and visualization...</h3>"))
    
    for key,value in best_results_fine.items():
        print(key, value)

    # compute and save refined structure - SELECT BEST Rw
    # Find the result with minimum Rw across all PDFs
    best_pdf = min(best_results_fine.items(), key=lambda x: x[1]['Rw'])
    pdf_file_best = best_pdf[0]
    result = best_pdf[1]
    
    # Accéder aux informations de structure et zoomscale
    strufile = result['strufile']
    zoomscale = result['zoomscale']
    
    print(f"\nBest result selected:")
    print(f"PDF: {os.path.basename(pdf_file_best)}")
    print(f"  Structure: {os.path.basename(strufile)}")
    print(f"  Rw: {result['Rw']:.4f}")
    print(f"  Zoomscale: {zoomscale:.6f}")
    
    # Appliquer la transformation de structure avec le zoomscale
    customizer = StructureCustom(strufile, zoomscale=zoomscale)
    stru = customizer.transform_structure()

    # Obtenir les informations de structure
    info = customizer.get_structure_info()
    print(f"Nombre d'atomes: {info['natoms']}")
    print(f"Composition: {info['composition']}")
    
    # Read fit results from saved .fit file instead of re-running refinement
    pdfname = os.path.basename(pdf_file_best).split('.')[0]
    struname = os.path.basename(strufile).split('.')[0]
    fit_filename = f"{pdfname}_{struname}.fit"
    fit_filepath = os.path.join(strufile_dir, 'fit', fit_filename)
    
    # Load fit file: # x  ycalc  y  dy
    r_calc, g_calc, g_exp, dy = np.loadtxt(fit_filepath, unpack=True)
    
    # write report
    report_gen = StructureReportGenerator(
        strufile_dir=strufile_dir,
        best_results= best_results_fine)

    # Extraire juste le nom du fichier sans le chemin complet
    pdf_basename = os.path.basename(pdf_file_best).replace('.gr', '')
    
    pdf_path = report_gen.generate_pdf_report(
        output_path=f'structure_screening_report_{pdf_basename}.pdf',
        n_top=10,
        pdf_file=pdf_file_best
    )

    print(f"\n✅ Report PDF generated successfully!")
    print(f"📄 Path: {pdf_path}")
    
    # Affichage final avec figure
    clear_output(wait=True)
    display(HTML("<h3>✅ ANALYSIS COMPLETE!</h3>"))
    
    # Create final figure: PDF fit only
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    # PDF refinement plot
    ax.plot(r_calc, g_exp, 'o', markersize=2, alpha=0.5, label='Experimental')
    ax.plot(r_calc, g_calc, '-', linewidth=2, label='Calculated')
    ax.plot(r_calc, g_exp[:len(r_calc)] - g_calc, '-', linewidth=1, alpha=0.5, label='Difference')
    ax.set_xlabel('r (Å)', fontsize=12)
    ax.set_ylabel('G(r)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'PDF Refinement (Rw = {result["Rw"]:.4f})', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Structure visualization using StructureCustom
    print("\n3D Structure Visualization:")
    view = customizer.view_structure(style='sphere', width=600, height=400, spin=True)
    display(view)
    
    print("\n" + "="*70)
    print(f"PDF analyzed: {os.path.basename(pdf_file_best)}")
    print(f"Best structure: {os.path.basename(strufile)}")
    print(f"Rw factor: {result['Rw']:.4f}")
    print(f"Zoomscale: {zoomscale:.6f}")
    print(f"Number of atoms: {info['natoms']}")
    print(f"Composition: {info['composition']}")
    print(f"\nReport saved: {pdf_path}")
    print(f"Results saved: {strufile_dir}/screening_results.json")
    print("="*70)
    
    # Clean up temporary directory
    try:
        shutil.rmtree(temp_pdf_dir)
    except Exception as e:
        print(f"\n⚠️  Warning: Could not clean up temporary directory {temp_pdf_dir}: {e}")

