# from socket import gethostname

line = 'oiii'




red = 0.01135 # redshift

orig_wave = 5008.239 # vac OIII

possigma = 3. # range in acceptable wavelength overlap [currently unimplemented]



platscl = 1.25 # Angstroms per spectral pixel



# [O III] line



start = 10 # beginning of spectrum in array space

end = 450 # end of spectrum in array space



low1 = 170 # beginning of left continuum segment in array space

upp1 = 200 # end of left continuum segment in array space



low2 = 270 # beginning of right continuum segment in array space

upp2 = 300 # end of right continuum segment in array space



plotmin = 5000 # minimum wavelength for x-axis in figures

plotmax = 5100 # maximum wavelength for x-axis in figures



#------------



minwave = 5054 # minimum wavelength for multinest prior

wave_range = 20.0 # wavelength prior space range (maxwave = minwave + wave_range)



minwidth = 1.67 # minimum line width for multinest prior ; R ~ 3000 @ 5008A

maxwidth = 33.83 # maximum line width for multinest prior ; 2000 km/s

maxwidth2 = 45.0



fluxsigma = 3. # minimum flux height for multinest prior (i.e. 3*sigma)



target = "ic5063"

basepath = '/home/dashtamirova/IC5063/ic5063'

