from functions.functions_fit_2D import master_fit_2D_alt_combined_shots, master_fit_2D_alt, master_fit_ne_Te_2D_quadratic


#master_fit_2D_alt_combined_shots(1030523030, 587, 700, smoothing_window=15)



#master_fit_2D_alt_combined_shots([1091210025, 1091210026, 1091210027, 1091210028], [608, 586, 582, 587], [700, 700, 700, 700], smoothing_window=15)

#master_fit_2D_alt_combined_shots([1091210025, 1091210026, 1091210027, 1091210028], [630, 586, 582, 587], [700, 700, 700, 700], smoothing_window=15)

#master_fit_2D_alt_combined_shots([1091210027], [582], [700], smoothing_window=15)

master_fit_ne_Te_2D_quadratic([1091210027, 1091210028], [582, 587], [700, 700])
