.. _faqs:

FAQ
===

Below you can find answers to commonly asked questions. If you don't see your answer
there you could also try posting on `stackoverflow <https://stackoverflow.com/>`_.

1. **What if I find a bug or have an idea for a new feature?**

    For bug reports or feature requests please use our
    `GitHub issue tracker <https://github.com/BCG-Gamma/facet/issues>`_.
    For any other enquiries please feel free to contact us at FacetTeam <at> bcg <dot> com.

2. **How does FACET's novel algorithm calculate feature redundancy and synergy?**

    Please keep an eye out for our publication coming soon. In the meantime please feel
    free to explore the GammaScope article (add link) to get a good introduction to
    using the algorithm.

3. **How can I contribute?**

    We welcome contributors! If you have minor changes in mind that would like to
    contribute, please feel free to create a pull request and be sure to follow the
    developer guidelines. For large or extensive changes please feel free to open an
    issue, or reach out to us at FacetTeam <at> bcg <dot> com to discuss.

4. **How can I perform standard plotting of SHAP values as done in the**
   `shap <https://github.com/slundberg/shap>`_ **library?**

    You can do this by creating an output of SHAP values from the fit LearnerInspector
    as in the example shown below.

    .. code-block:: Python

        # run inspector
        inspector = LearnerInspector(
            n_jobs=-3,
            verbose=False,
        ).fit(crossfit=ranker.best_model_crossfit)

        # get shap values and associated data
        shap_data = inspector.shap_plot_data()
        shap.summary_plot(shap_values=shap_data.shap_values, features=shap_data.features)



5. **How can I extract CV performance metrics from the LearnerRanker to create my
   own summaries or figures?**

    You can extract the desired information as a data frame from the fitted
    LearnerRanker object.

    .. code-block:: Python

        # after fitting a ranker
        cv_result_df = ranker.summary_report()

Citation
--------
If you use FACET in your work please cite us as follows:

Bibtex entry::

     @manual{
     title={FACET},
     author={FACET Team at BCG Gamma},
     year={2020},
     note={Python package version 1.0.0)
     }
