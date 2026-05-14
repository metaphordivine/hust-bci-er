# Route Board

generated_at: 2026-05-14

| route_id | status | family | decision |
|---|---|---|---|
| `conformer_srfnet_score_average` | IDEA | score fusion | average of conformer and srfnet reference scores |
| `conformer_srfnet_whitening_eps3e4_average` | IDEA | whitening score fusion | average of conformer and srfnet whitening eps3e4 scores |
| `conformer_srfnet_whitening_eps1e3_average` | IDEA | whitening score fusion | average of conformer and srfnet whitening eps1e3 scores |
| `conformer_srfnet_whitening_eps1e3_handcrafted_weight25_average` | IDEA | handcrafted score fusion | weighted average with 25 percent handcrafted-feature score weight |
| `conformer_srfnet_whitening_eps1e3_handcrafted_weight30_average` | IDEA | handcrafted score fusion | weighted average with 30 percent handcrafted-feature score weight |
| `conformer_srfnet_whitening_eps1e3_handcrafted_weight40_average` | IDEA | handcrafted score fusion | weighted average with 40 percent handcrafted-feature score weight |
| `whitening_eps1e3_with_conformer_srfnet_reference_average` | IDEA | score fusion | average combining whitening eps1e3 scores with conformer and srfnet reference scores |
| `whitening_eps1e3_query_eps3e4_srfnet_context_fusion` | IDEA | score fusion | query-context fusion using whitening eps1e3 as query and eps3e4/srfnet reference as context |
| `ea_deformer` | IDEA | EA + Deformer | scaffold example |
