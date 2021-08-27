# -*- coding: utf-8 -*-
#!/usr/bin/env python
import texttable

def show_test_list(test_list):
    table = [['TC', 'Description', 'model', 'Command']]
    for i in list(test_list):
        tc = str(i)
        description = test_list[i].get_description()
        command_dict = test_list[i].get_command()

        for model, cmd_set in command_dict.items():
            table.append([tc, description, model, cmd_set['command']])
            tc = ''
            description = ''

    t = texttable.Texttable()
    t.set_max_width(200)
    t.add_rows(table)
    print(t.draw())


def show_sqe_test_list(test_list):
    table = [['TC', 'Description', 'Command']]
    for i in list(test_list):
        tc = str(i)
        description = test_list[i].get_description()
        command = test_list[i].get_command()
        table.append([tc, description, command])

    t = texttable.Texttable()
    t.set_max_width(200)
    t.add_rows(table)
    print(t.draw())
