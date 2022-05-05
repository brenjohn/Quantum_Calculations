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

"""
For each xi in `bins`, computes the average y-value of points whose corresponding x-value is 
in a bin centred at xi and has width `bin_width`.

Returns arrays containing the averages and locations for non-empty bins.
"""
function coarse_grained_average(ys::Vector{T}, xs, bins, bin_width) where T <: Number
    average = zeros(T, length(bins))

    for (i, xi) in enumerate(bins)
        f = x -> (xi - bin_width/2) < x < (xi + bin_width/2)
        inds = findall(f, xs)
        average[i] = sum(ys[inds]) / length(inds)
    end

    inds = findall(!isnan, average)
    average[inds], bins[inds]
end