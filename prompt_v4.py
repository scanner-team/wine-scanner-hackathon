PROMPT_V4 = """Extract ONLY the text visible on this wine label into strict JSON.

Field definitions (do not confuse them):
- visible_name: the main product/brand name printed on the label. NOT the grape, NOT the producer.
- producer: the winery/company name that made the wine (e.g. "Кубань-Вино", "Абрау-Дюрсо", "Скалистый Берег"). NOT the grape variety.
- cuvee: a special series/line name WITHIN the producer's range (e.g. "Reserve", "Selection", "Limited Edition"). NOT the grape variety, NOT the vintage year, NOT the category. If the label has no separate series name beyond the main product name, use null.
- vintage: the 4-digit HARVEST year of THIS wine (e.g. "2024", "2023"). Extract ONLY the digits, never a label word like "ГОД УРОЖАЯ"/"vintage" itself.
  IMPORTANT: labels sometimes also show an unrelated year such as the WINERY'S FOUNDING year (often marked "осн." / "est." / "founded", e.g. "осн. 2010") — that is NOT the vintage, ignore it for this field. The vintage is usually printed near the bottom of the label, close to the wine style/category text, and is the most recent-looking year on the label.
- grape: the grape VARIETY name (e.g. "Мерло", "Каберне Совиньон", "Шардоне", "Ркацители") — a specific botanical grape cultivar name.
  IMPORTANT: the wine style/category text (e.g. "сухое красное", "полусладкое белое", "брют", "dry red") is NEVER a grape variety — if you only see a style/category phrase and no actual grape cultivar name, grape MUST be null. Do not copy the category text into this field.
- category: the wine style/type (e.g. "сухое красное", "полусладкое белое", "брют"). NOT country of origin, NOT other legal text.
- raw_visible_text: array of ALL text fragments visible on the label, verbatim, regardless of whether they fit the fields above. This field is MANDATORY, always include it even if short.

Rules:
- If a field's value is not clearly visible on the label, use null. Do not guess, do not use outside knowledge about wines.
- Each field must contain ONLY the type of information defined above.
- grape and category must never contain the same text as each other.
- Output strict JSON only, no explanations, no markdown code fences.
"""
