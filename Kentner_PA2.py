# UTILITY FILE: Contains the class definition for the node objects and the CSP/backtracking functions
# Author: Kyle Kentner
# Instructor: Dr. Doug Heisterkamp
# Modification of PA2 (Constraint Satisfaction and Backtracking)

#######################
# NODE CLASS DEFINITION
class Node:
    # domain = set()
    similarity_index = 0.9
    location = []
    incoming = []
    rgb = []


####################
# PRINTING FUNCTIONS
# Prints the possible values of each node in the CSP (csp argument is a 1-D queue)
def print_csp_queue(csp, sz):
    output_string = ''
    col = 1
    for c in csp:
        output_string += str(c.domain) + ' '
        if col % sz == 0:
            output_string += '\n'
        col += 1

    print(output_string)


# Prints the possible values of each node in the CSP (csp argument is a 2-D array)
def print_csp(csp):
    x = 0
    output_string = ''
    while x < len(csp):
        y = 0
        while y < len(csp[x]):
            output_string += str(csp[x][y].domain) + ' '
            y += 1
        output_string += '\n'
        x += 1

    print(output_string)


###############
# CSP FUNCTIONS
# Copies and returns a set of domain values
def copy_domain(dm):
    out_domain = set()
    for d in dm:
        out_domain.add(d)

    return out_domain


# Copies the CSP's values into a queue for processing; the CSP returned is completely detached from the original
def copy_csp_queue_complete(csp, sz):
    out_queue = []
    for c in csp:
        nd = Node()
        nd.incoming = c.incoming
        nd.location = c.location

        # Add the domain values individually to break the references
        nd.domain = set()
        for d in c.domain:
            nd.domain.add(d)

        out_queue.append(nd)

    # Update incoming to use the new references
    for o in out_queue:
        i = 0
        while i < len(o.incoming):
            o.incoming[i] = out_queue[o.incoming[i].location[0] * sz + o.incoming[i].location[1]]
            i += 1

    return out_queue


# Copies the CSP's values into a queue for processing; domains are left as references to the parent domains
def copy_csp_queue(csp):
    out_queue = []
    for c in csp:
        nd = Node()
        nd.domain = c.domain
        nd.incoming = c.incoming
        nd.location = c.location
        out_queue.append(nd)

    return out_queue


# Checks a node for conflicts and removes elements from the domain as necessary; returns True/False based on whether
#   revisions were made
def revise(node_a, node_b):
    revision = False
    removal_list = []

    # Check each value in node_a's domain
    for a in node_a.domain:
        # Assume a local revision will occur
        local_revision = True

        # Check each value in node_b's domain
        for b in node_b.domain:
            # If some value in node_b's domain satisfies the constraint, no revision is necessary
            if a != b:
                local_revision = False
                break

        # If any revisions occurred for this value, add the value to a list
        if local_revision:
            removal_list.append(a)
            revision = True

    # Remove whatever invalid values are in the domain
    for v in removal_list:
        # The value will also be removed from the corresponding node in the original CSP queue
        node_a.domain.remove(v)

    return revision


# Iterates over all nodes in the CSP queue and checks for conflicts; narrows the domain based on constraints; returns
#   True if a solution can be constructed and prints the solution; returns False if the domain of some node is
#   completely emptied
# Based on Russel and Norvig, p. 186
def AC3(csp, sz, csp_out):
    # Create new nodes based on the original CSP; domains will update concurrently, but csp's nodes will not be removed
    node_queue = copy_csp_queue(csp)

    # While the queue has nodes to investigate, perform the AC-3 algorithm
    while len(node_queue) > 0:
        # Dequeue a node
        current = node_queue.pop(0)

        # Check each node in current.incoming
        index = 0
        while index < len(current.incoming):
            # Get the first neighbor from the incoming list
            first_neighbor = current.incoming[index]

            # Check if current's domain must be revised,
            if revise(current, first_neighbor):
                # Check if all options have been exhausted
                if len(current.domain) == 0:
                    return False

                # Re-investigate all other nodes since current's domain has been revised
                for other_node in current.incoming:
                    # Add neighbors back to the queue
                    if other_node != first_neighbor:
                        node_queue.append(other_node)
            index += 1

    # Print the CSP after successful termination
    if csp_out:
        print_csp_queue(csp, sz)

    return True


########################
# BACKTRACKING FUNCTIONS
# Select an unassigned node based on the minimum remaining values heuristic
def select_unassigned_node(csp, assignment):
    min_node = None
    min_remaining = 1000000

    # Use the node locations as unique identifiers (references to csp's Node objects will differ from assignment's)
    assigned_coords = []
    for var in assignment:
        assigned_coords.append(var.location)

    # Check each node in the CSP
    for node in csp:
        # Update the node with the fewest options in its domain as needed
        if node.location not in assigned_coords and len(node.domain) < min_remaining:
            min_node = node
            min_remaining = len(node.domain)

    return min_node


# Assign a value to the node with the minimum remaining values and infer dependent nodes using AC-3;
#   return the final assignment and print the CSP's solution if it is found
# Based on Russel and Norvig, p. 191 and University of Toronto
#   http://www.cs.toronto.edu/~hojjat/384w09/Lectures/Lecture-04-Backtracking-Search.pdf
def backtrack(csp, sz, assignment, fe_ct):
    # Check if the assignment is complete
    if len(assignment) == len(csp):
        # Success; values have been assigned to all nodes
        print_csp_queue(csp, sz)
        print('For-each iterations:', fe_ct)
        return assignment

    # Copy the CSP to save the start state
    start_csp = copy_csp_queue_complete(csp, sz)
    cur_csp = copy_csp_queue_complete(csp, sz)

    # The assignment is incomplete; select an unassigned Node
    node = select_unassigned_node(cur_csp, assignment)

    # Add the node to assignment
    assignment.add(node)

    # Check each value in the node's domain and see if it is consistent with assignment
    init_domain = copy_domain(node.domain)
    for val in init_domain:
        # Assign val and perform the AC-3 algorithm
        node.domain = {val}
        inferences = AC3(cur_csp, sz, False)

        # Check if AC-3 succeeded
        if inferences != False:
            # Continue the backtracking algorithm
            result = backtrack(cur_csp, sz, assignment, fe_ct)
            if len(result) == len(csp):
                break
        else:
            # Remove the node from assignment and reset the CSP
            assignment.remove(node)
            cur_csp = copy_csp_queue_complete(start_csp, sz)
            fe_ct += 1

    return result


# Driver for the recursive backtrack function; sends a blank set as the initial assignment
# Based on Russel and Norvig, p. 191 and University of Toronto
#   http://www.cs.toronto.edu/~hojjat/384w09/Lectures/Lecture-04-Backtracking-Search.pdf
def backtracking_search(csp, sz, foreach_count):
    # Backtrack using the provided csp and a blank domain
    return backtrack(csp, sz, set(), foreach_count)


##########################
# INITIALIZATION FUNCTIONS
# Initializes the CSP using the problem size, the subdivision size, and the user's input-file values
def init_csp(sz, subdiv, fixed):
    # Create a 2-D array of nodes
    node_list = []

    # Maintain a list of top-left node coordinates based on the subdivision size
    corners = []

    # Row index
    i = 0
    while i < sz:
        # List of preceding nodes for the row and column
        cur_row = []

        # Column index
        j = 0
        while j < sz:
            # Instantiate a new Node; initialize location and domain
            cur_node = Node()
            cur_node.location = [i, j]

            # Generalization; domain is based on the problem size
            if sz == 9:
                cur_node.domain = {1, 2, 3, 4, 5, 6, 7, 8, 9}
            elif sz == 3:
                cur_node.domain = {1, 2, 3}

            # Add to the list
            cur_row.append(cur_node)

            # If this j is a top-left corner, add (i, j) to the list
            if i % subdiv == 0 and (j - i) % subdiv == 0:
                corners.append([i, j])

            j += 1

        # Add the row to the 2D array
        node_list.append(cur_row)

        i += 1

    # Initialize the incoming lists for the rows and columns
    m = 0
    n = 0
    for row in node_list:
        for node in row:
            # Get the coordinates of the current node
            i = node.location[0]
            j = node.location[1]

            # Get the top-left corner of the associated sector
            m = corners[int(j / subdiv) + int(i / subdiv) * subdiv][0]
            n = corners[int(j / subdiv) + int(i / subdiv) * subdiv][1]

            # Get each of the nodes which has an arc that arrives at the current node
            node.incoming = []
            k = 0
            while k < sz:
                # ROWS AND COLUMNS
                # Add each other element in the row to this node's incoming list
                if k != j:
                    node.incoming.append(node_list[i][k])

                # Add each other element in the column to this node's incoming list
                if k != i:
                    node.incoming.append(node_list[k][j])

                    # Generalization; the 9x9 problem does not require a unique forward diagonal
                    if subdiv == 1 and i == j:
                        node.incoming.append(node_list[k][k])

                # Generalization; the 3x3 problem does not account for the Sudoku subdivisions
                if subdiv > 1:
                    # SECTORS
                    p = m
                    q = n + k % subdiv

                    # Add each other element in the sector to this node's incoming list
                    if p != i and q != j:
                        node.incoming.append(node_list[p][q])

                # Increment k
                k += 1

                # Increment m if k exceeds the subdivision bounds
                if k % subdiv == 0:
                    m += 1

    # Initialize the nodes specified in the user's input file
    while len(fixed) > 0:
        # Dequeue the first element and update the appropriate node
        cur_pa = fixed.pop(0)
        node_list[cur_pa[0]][cur_pa[1]].domain = {cur_pa[2]}

    # Convert the 2-D array into a 1-D queue
    Q = []
    for row in node_list:
        for node in row:
            Q.append(node)

    # Return the CSP queue
    return Q


# Parses the user's input file and creates a list of coordinates/fixed values
def parse_file(f_name):
    fixed_coords = []

    # Open the file
    with open(f_name, mode='r') as in_file:
        while True:
            # Read and parse each line of the file
            ln = in_file.readline()

            # If at the end of the file
            if not ln:
                break

            # Separate ln on spaces
            vals = ln.split(sep=' ')
            if len(vals) == 3:
                fixed_coords.append([int(vals[0]), int(vals[1]), int(vals[2])])

    return fixed_coords


# Gets the file name based on the user's command-line input
def get_file_name(sys_args):
    # Parse initialization data from the file given via the command line
    arg = 1
    while arg < len(sys_args):
        if arg < 2:
            # Assume this argument is the file name
            f_name = sys_args[arg]
        else:
            # Assume the file path has spaces if more than two args
            f_name += ' ' + sys_args[arg]

        arg += 1

    return f_name