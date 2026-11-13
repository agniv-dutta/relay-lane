export default function SpecimenView() {
  const stretches = [70, 100, 140];
  return (
    <div>
      <h1>Type specimen</h1>
      {stretches.map((s) => (
        <div key={s} style={{ marginBottom: "2rem" }}>
          <div style={{ "--stretch": s } as any}>
            <h2 style={{ fontVariationSettings: `"wdth" ${s}, "wght" 700` }}>relay-lane at stretch {s}</h2>
            <p style={{ fontFamily: "var(--font-body)", maxWidth: "66ch", lineHeight: 1.6 }}>
              The quick brown fox jumps over the lazy dog. Source Serif 4 body text for passages from the papers.
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}
