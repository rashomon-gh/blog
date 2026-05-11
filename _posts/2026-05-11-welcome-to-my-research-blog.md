---
layout: post
title: "Welcome to my Research Blog"
date: 2026-05-11 12:00:00 -0000
author: Shawon
---

Welcome to my academic and research blog. This space is designed to be minimal, allowing the content to take center stage. Below are some examples of what this platform supports.

## Typography

The body text uses **Noto Sans**, which provides a clean and highly readable experience, making it perfect for long-form research notes and academic articles. The layout is optimized to be responsive and distraction-free.

<!-- excerpt -->

## Mathematics with KaTeX

Mathematics rendering is crucial for academic writing. This blog uses KaTeX to render LaTeX equations beautifully and quickly across all modern browsers.

Here is an inline equation: Let $f(x) = x^2$ be a function defined on the real numbers. The derivative of this function is $f'(x) = 2x$. 

And here is a multiline block equation:

$$
\begin{aligned}
\int_0^\infty e^{-x^2} dx &= \frac{\sqrt{\pi}}{2} \\
E &= mc^2
\end{aligned}
$$

Alternatively, you can use standard `$$ ... $$` for display math:

$$
\mathcal{L} = \bar{\psi} (i\gamma^\mu \partial_\mu - m) \psi
$$

## Code Snippets

Code blocks are styled with **Fira Code**, a popular monospaced font tailored for programming. Syntax highlighting is handled by Rouge, providing automatic line numbers.

Here is a quick Python script to demonstrate:

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_fourier_transform(signal: np.ndarray) -> np.ndarray:
    """
    Computes the Fast Fourier Transform (FFT) of a 1D signal.
    """
    fft_result = np.fft.fft(signal)
    return np.abs(fft_result)

if __name__ == "__main__":
    t = np.linspace(0, 1, 500)
    signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
    
    spectrum = compute_fourier_transform(signal)
    print(f"Computed spectrum with {len(spectrum)} points.")
```

## Conclusion

This concludes the demonstration post. You can continue adding your research thoughts, equations, and code to this platform easily!
