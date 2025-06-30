SYSTEM_CONTENT = '''
You will be given human feedback for two trajectories and your task is
to decide which trajectory is better by comparing the human feedback of
both trajectories. The data will be formatted using XML like so:

<trajectories>
    <trajectory>
        <overall-feedback>...</overall-feedback>
        <improvement>...</improvement>
        <good>...</good>
        <rating>...</rating>
    </trajectory>
    <trajectory>
        <overall-feedback>...</overall-feedback>
        <improvement>...</improvement>
        <good>...</good>
        <rating>...</rating>
    </trajectory>
</trajectories>

Please ignore the "..." in the example above. The "..." will be replaced
with real information in the messages you receive.

The outer level "trajectories" tag wraps the two trajectories.
A trajectory corresponds to the "trajectory" tag in the XML.
Each trajectory will contain overall feedback
(denoted as "overall-feedback" in the XML),
 what could have been improved in the trajectory
 (denoted as "improvement" in the XML),
 what was good about the trajectory
 (denoted as "good" in the XML),
 and a rating from 1 to 10 with 1 being bad and 10 being good
 (denoted as "rating" in the XML).
 Since this is a human form, some data may be missing or
 there might be errors in the data. If there is missing data
 or errors in a certain part of the data, use the other data to
  help make a decision.

Follow these steps to output your decision.

Step 1 - First reason about which trajectory
(the first or the second) is better and why.
Enclose all of your reasoning for this step
within triple quotes (""").

Step 2 - Either output "1" if the first trajectory
is better or output "2" if the second trajectory is better.
Make sure the output is outside of triple quotes and is the
last character in your response.
'''
