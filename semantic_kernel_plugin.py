"""Wraps this project's real NER/contact-extraction/ranking functions
(main.py's find_subject, extract_contact, infer_companions,
rank_messages) as a Semantic Kernel native plugin, orchestrated through
a real Kernel instance. No LLM connector is registered or required --
this exercises Semantic Kernel's plugin/function-invocation
architecture directly (kernel.invoke against native functions), the
same mechanism a chat-completion-driven planner would call into.

Run it directly:
    python semantic_kernel_plugin.py "did Sophia Al-Farsi book a flight?"
"""

import asyncio
import sys

from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function

import main as qa


class MessageInsightsPlugin:
    """Native Semantic Kernel functions over the real message dataset
    already loaded and embedded by main.py at import time."""

    @kernel_function(description="Finds which known user a question is about, via NER or first-name match.")
    def find_subject(self, question: str) -> str:
        subject = qa.find_subject(question)
        return subject or "unknown"

    @kernel_function(description="Extracts any email addresses or phone numbers mentioned in a message.")
    def extract_contact(self, text: str) -> str:
        contacts = qa.extract_contact(text)
        return ", ".join(contacts) if contacts else "none found"

    @kernel_function(description="Infers whether the message implies the person is traveling with companions.")
    def infer_companions(self, message: str) -> str:
        return qa.infer_companions(message)

    @kernel_function(description="Ranks this user's real messages by semantic similarity to the question, most relevant first.")
    def rank_top_message(self, question: str, user_name: str) -> str:
        all_messages = qa.df.to_dict("records")
        user_messages = [m for m in all_messages if m.get(qa.user_col) == user_name] if qa.user_col else all_messages
        ranked = qa.rank_messages(question, user_messages)
        if not ranked:
            return "no messages found"
        return qa.clean_text(ranked[0][qa.msg_col])


async def run(question: str) -> None:
    kernel = Kernel()
    plugin = kernel.add_plugin(MessageInsightsPlugin(), plugin_name="MessageInsights")

    subject_result = await kernel.invoke(plugin["find_subject"], question=question)
    subject = str(subject_result)
    print(f"[find_subject] -> {subject}")

    if subject != "unknown":
        top_message_result = await kernel.invoke(plugin["rank_top_message"], question=question, user_name=subject)
        top_message = str(top_message_result)
        print(f"[rank_top_message] -> {top_message}")

        companions_result = await kernel.invoke(plugin["infer_companions"], message=top_message)
        print(f"[infer_companions] -> {companions_result}")

        contact_result = await kernel.invoke(plugin["extract_contact"], text=top_message)
        print(f"[extract_contact] -> {contact_result}")


def main() -> None:
    question = " ".join(sys.argv[1:]) or "did Sophia Al-Farsi book a flight?"
    print(f"Question: {question!r}\n")
    asyncio.run(run(question))


if __name__ == "__main__":
    main()
