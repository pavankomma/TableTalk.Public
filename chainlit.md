# TableTalk

Ask questions about your database in plain English. TableTalk retrieves the
relevant schema cards, business notes, and example queries from its vector
feed, writes a read-only SQL query, runs it, and answers from the rows. The
generated SQL is shown above every answer.

If the first attempt looks unreliable (an error, zero rows), TableTalk
retries with more reasoning and then hands the question to a tool-calling
agent that explores the schema itself. The step header shows which rung
answered.

Try a starter, or ask anything about the data. Follow-up questions work,
because the last few turns are carried as context. When an answer is
right, click **👍 Save as example** to teach TableTalk.

To browse and edit the feed, start the API server and open the
[feed editor](http://localhost:8001/feed).
