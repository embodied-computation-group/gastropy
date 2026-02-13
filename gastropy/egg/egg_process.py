"""EGG signal processing pipeline."""


def egg_process(signal, sampling_rate, method="default"):
    """Process an electrogastrography (EGG) signal.

    Applies a full processing pipeline to a raw EGG signal, including
    filtering, artifact removal, and gastric rhythm extraction.

    Parameters
    ----------
    signal : array_like
        The raw EGG signal as a 1D array or pandas Series.
    sampling_rate : int or float
        The sampling rate of the signal in Hz.
    method : str, optional
        The processing method to use. Default is ``"default"``.

    Returns
    -------
    signals : pd.DataFrame
        A DataFrame containing the processed signal and extracted
        components (e.g., filtered signal, gastric phase, amplitude).
    info : dict
        A dictionary containing metadata and intermediate processing
        results (e.g., filter parameters, detected cycles).

    See Also
    --------
    egg_clean : Clean an EGG signal (filtering and artifact removal).
    egg_analyze : Extract metrics from a processed EGG signal.

    Examples
    --------
    >>> import gastropy as gp
    >>> signals, info = gp.egg_process(raw_signal, sampling_rate=256)
    """
    raise NotImplementedError("egg_process is not yet implemented.")
