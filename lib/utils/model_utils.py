def format_token_init_types(token_init_types) -> str:
    def _format_numeric(v: float) -> str:
        # Convert scientific notation: 1e-3 -> 1em3
        # For small floats (< 1), format as scientific notation
        if isinstance(v, float) and v < 1 and v > 0:
            s = f"{v:.0e}"
            if 'e-' in s:
                base, exp = s.split('e-')
                exp = str(int(exp))  # Remove leading zeros
                return f"{base}m{exp}"
            if 'e+' in s:
                base, exp = s.split('e+')
                exp = str(int(exp))  # Remove leading zeros
                return f"{base}p{exp}"
            return s

        # For integers or larger floats, use string representation
        return str(v).replace('.', '')

    parts = []
    for t in token_init_types:
        if isinstance(t, (int, float)):
            parts.append(_format_numeric(float(t)))
        else:
            name = str(t).strip()
            lower = name.lower()

            # Special-cases for global token specs: fix1e-4 / lrn1e-4 (or with ':'/'_')
            parsed = False
            for canonical_prefix in ("lrn", "fix"):
                if not lower.startswith(canonical_prefix):
                    continue
                rest = lower[len(canonical_prefix):].lstrip(":_")
                if not rest:
                    rest = "1e-4"
                try:
                    parts.append(f"{canonical_prefix}{_format_numeric(float(rest))}")
                    parsed = True
                except ValueError:
                    parsed = False
                break

            if not parsed:
                parts.append(name)
    return ''.join(parts)
