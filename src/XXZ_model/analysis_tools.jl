using LsqFit

"""
Performs an unfolding procedure, similar to the one described in Santos_2010,
on the given energy spectrum.  
"""
function unfold_spectrum(spectrum)
    staircase(E) = sum(En -> En <= E, spectrum)
    domain = range(start = spectrum[1], stop = spectrum[end], length=2*length(spectrum))
    image = staircase.(domain)

    model(x, params) = sum(i -> x.^(i-1) .* params[i], 1:length(params))
    params = zeros(15)
    best_fit = curve_fit(model, domain, image, params)
    unfolded_spectrum = model(spectrum, coef(best_fit))

    unfolded_spectrum
end