import asyncio

from semantic_kernel import Kernel

from semantic_kernel_plugin import MessageInsightsPlugin


def test_find_subject_identifies_a_real_known_user():
    kernel = Kernel()
    plugin = kernel.add_plugin(MessageInsightsPlugin(), plugin_name="MessageInsights")

    result = asyncio.run(kernel.invoke(plugin["find_subject"], question="did Sophia Al-Farsi book a flight?"))
    assert str(result) == "Sophia Al-Farsi"


def test_end_to_end_pipeline_returns_a_real_ranked_message():
    kernel = Kernel()
    plugin = kernel.add_plugin(MessageInsightsPlugin(), plugin_name="MessageInsights")

    subject = str(asyncio.run(kernel.invoke(plugin["find_subject"], question="did Sophia Al-Farsi book a flight?")))
    assert subject != "unknown"

    top_message = str(
        asyncio.run(
            kernel.invoke(plugin["rank_top_message"], question="book a flight", user_name=subject)
        )
    )
    assert "flight" in top_message.lower() or "jet" in top_message.lower()
