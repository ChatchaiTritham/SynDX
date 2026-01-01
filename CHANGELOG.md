# What's Changed in SynDX

Hey, this file tracks what we've built, fixed, and changed over time.

## Version 0.1.0 (January 2025)

### What's New

So this is our first public release! Here's what made it in:

**The Three-Phase Pipeline**
- Built out Phase 1 to extract clinical knowledge from TiTrATE guidelines
- Phase 2 brings together NMF, VAE, SHAP, and counterfactual generation
- Phase 3 handles validation across stats, diagnostics, and XAI metrics

**Healthcare Standards Integration**
- Got HL7 FHIR R4 export working so you can actually use this data
- Mapped everything to SNOMED CT and LOINC codes

**Developer Experience**
- Docker containers so setup isn't a nightmare
- Wrote docs and tutorials (hopefully they're helpful!)

### Quick Heads-Up

Look, this is research code from a PhD project. We haven't validated any of this against real patients yet. All our numbers come from synthetic-to-synthetic comparisons, which means they show internal consistency but NOT clinical utility. Don't use this for actual patient care without proper validation studies.

---

*Need the formal changelog? Check git history for detailed commit messages.*
